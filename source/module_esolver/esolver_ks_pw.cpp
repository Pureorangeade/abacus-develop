#include "esolver_ks_pw.h"

#include <iostream>

#include "module_io/nscf_band.h"
#include "module_io/write_dos_pw.h"
#include "module_io/write_istate_info.h"
#include "module_io/write_wfc_pw.h"
#include "module_io/output_log.h"
#include "module_io/input_conv.h"

//--------------temporary----------------------------
#include "module_elecstate/module_charge/symmetry_rho.h"
#include "module_elecstate/occupy.h"
#include "module_hamilt_general/module_ewald/H_Ewald_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/print_info.h"
//-----force-------------------
#include "module_hamilt_pw/hamilt_pwdft/forces.h"
//-----stress------------------
#include "module_hamilt_pw/hamilt_pwdft/stress_pw.h"
//---------------------------------------------------
#include "module_base/memory.h"
#include "module_elecstate/elecstate_pw.h"
#include "module_hamilt_general/module_vdw/vdw.h"
#include "module_hamilt_pw/hamilt_pwdft/hamilt_pw.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_hsolver/hsolver_pw.h"
#include "module_hsolver/kernels/dngvd_op.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_io/berryphase.h"
#include "module_io/numerical_basis.h"
#include "module_io/numerical_descriptor.h"
#include "module_io/rho_io.h"
#include "module_io/potential_io.h"
#include "module_io/to_wannier90_pw.h"
#include "module_io/winput.h"
#include "module_io/write_wfc_r.h"
#include "module_base/module_device/device.h"
//---------------------------------------------------
#include "module_psi/psi_initializer_atomic.h"
#include "module_psi/psi_initializer_nao.h"
#include "module_psi/psi_initializer_random.h"
#include "module_psi/psi_initializer_atomic_random.h"
#include "module_psi/psi_initializer_nao_random.h"
//---------------------------------------------------
#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif
#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::ESolver_KS_PW()
{
    this->classname = "ESolver_KS_PW";
    this->basisname = "PW";
    this->device = base_device::get_device_type<Device>(this->ctx);
#if ((defined __CUDA) || (defined __ROCM))
    if (this->device == base_device::GpuDevice)
    {
        hsolver::createGpuBlasHandle();
        hsolver::createGpuSolverHandle();
        container::kernels::createGpuBlasHandle();
        container::kernels::createGpuSolverHandle();
    }
#endif
}

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::~ESolver_KS_PW()
{
    // delete HSolver and ElecState
    if (this->phsol != nullptr)
    {
        delete reinterpret_cast<hsolver::HSolverPW<T, Device>*>(this->phsol);
        this->phsol = nullptr;
    }
    if (this->pelec != nullptr)
    {
        delete reinterpret_cast<elecstate::ElecStatePW<T, Device>*>(this->pelec);
        this->pelec = nullptr;
    }
    // delete Hamilt
    if (this->p_hamilt != nullptr)
    {
        delete reinterpret_cast<hamilt::HamiltPW<T, Device>*>(this->p_hamilt);
        this->p_hamilt = nullptr;
    }
    if (this->device == base_device::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        hsolver::destoryBLAShandle();
        hsolver::destroyGpuSolverHandle();
        container::kernels::destroyGpuBlasHandle();
        container::kernels::destroyGpuSolverHandle();
#endif
        delete reinterpret_cast<psi::Psi<T, Device>*>(this->kspw_psi);
    }
    if (GlobalV::precision_flag == "single")
    {
        delete reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->__kspw_psi);
    }
    delete this->psi;
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::Init_GlobalC(Input& inp, UnitCell& cell)
{
    // GlobalC is a historically left-over namespace, it is used to store global classes,
    // including:
    // pseudopot_cell_vnl: pseudopotential in cell, V non-local
    // UnitCell: cell information with atomic properties
    // Grid_Driver: 
    // Parallel_Grid: 
    // Parallel_Kpoints:
    // Restart:
    // Exx_Info:
    // Exx_Lip:

    // GlobalC would be refactored out in the future. If there is better idea about how
    // to organize information stored in classes above, please feel free to discuss with
    // issue or pull request.

    if (this->psi != nullptr)
    {
        delete this->psi;
    }

    // allocate memory for std::complex<double> datatype psi
    // New psi initializer in ABACUS, Developer's note:
    // Because the calling relationship between ESolver_KS_PW and derived class is
    // complicated, up to upcoming of ABACUS 3.4, we only implement this new psi
    // initialization method for ksdft_pw, which means the routinely used dft theory.
    // For other theories like stochastic DFT, we still use the old method.

    // LCAOINPW also temporarily uses ESolver_KS_PW workflow, but in principle, it
    // should have its own ESolver. ESolver class is for controlling workflow for each
    // theory-basis combination, in the future it is also possible to seperate/decouple
    // the basis (representation) with operator (hamiltonian) and solver (diagonalization).
    // This feature requires feasible Linear Algebra library in-built in ABACUS, which
    // is not ready yet.
    if (GlobalV::psi_initializer) // new method
    {
        // psi_initializer drag initialization of pw wavefunction out of HSolver, make psi
        // initialization decoupled with HSolver (diagonalization) procedure.
        // However, due to EXX is hard to maintain, we still use the old method for EXX.
        // LCAOINPW in version >= 3.5.0 uses this new method.
        this->psi = this->psi_init->allocate();
    }
    else // old method
    {
        // old method explicitly requires variables such as total number of kpoints, number
        // of bands, number of G-vectors, and so on. Comparatively in new method, these 
        // variables are imported in function called initialize.
        this->psi = this->wf.allocate(this->kv.nkstot, this->kv.nks, this->kv.ngk.data(), this->pw_wfc->npwk_max);
    }
    // ---------------------------------------------------------------------------------

    //! init pseudopotential
    GlobalC::ppcell.init(GlobalC::ucell.ntype, &this->sf, this->pw_wfc);

    //! initalize local pseudopotential
    GlobalC::ppcell.init_vloc(GlobalC::ppcell.vloc, this->pw_rhod);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "LOCAL POTENTIAL");

    //! Initalize non-local pseudopotential
    GlobalC::ppcell.init_vnl(GlobalC::ucell, this->pw_rhod);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "NON-LOCAL POTENTIAL");

    // ---------------------------------------------------------------------------------
    // there is a strange bug that the following procedure must be placed here, otherwise
    // the value of psi will finally (before diagonalization) be all-zero, still dont know
    // why.
    if (!GlobalV::psi_initializer)
    {
        // however, init_at_1 does not actually initialize the psi, instead, it is a
        // function to calculate a interpolate table saving overlap intergral or say
        // Spherical Bessel Transform of atomic orbitals.
        this->wf.init_at_1(&this->sf);
        // similarly, wfcinit not really initialize any wavefunction, instead, it initialize
        // the mapping from ixy, the 1d flattened index of point on fft grid (x, y) plane,
        // to the index of "stick", composed of grid points.
        this->wf.wfcinit(this->psi, this->pw_wfc);
    }
    // ---------------------------------------------------------------------------------


    this->kspw_psi = GlobalV::device_flag == "gpu" 
                         || GlobalV::precision_flag == "single"
                         ? new psi::Psi<T, Device>(this->psi[0])
                         : reinterpret_cast<psi::Psi<T, Device>*>(this->psi);

    // I would like to change the above sentence to the following, 
    // but I am not sure what the code is doing, so I leave it as a comment
    // mohan by 2024-03-27
/*
	if (GlobalV::device_flag == "gpu" || GlobalV::precision_flag == "single") 
	{
        // psi[0] means gamma_only?
		this->kspw_psi = new psi::Psi<T, Device>(this->psi[0]);
	} 
	else 
	{
		this->kspw_psi = reinterpret_cast<psi::Psi<T, Device>*>(this->psi);
	}
*/

    if (GlobalV::precision_flag == "single")
    {
        ModuleBase::Memory::record("Psi_single", sizeof(T) * this->psi[0].size());
    }

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT BASIS");
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_all_runners(Input& inp, UnitCell& ucell)
{
    // 1) call before_all_runners() of ESolver_KS
    ESolver_KS<T, Device>::before_all_runners(inp, ucell);

    // 2) initialize HSolver
    if (this->phsol == nullptr)
    {
        this->phsol = new hsolver::HSolverPW<T, Device>(this->pw_wfc, &this->wf);
    }

    // 3) initialize ElecState,
    if (this->pelec == nullptr)
    {
        this->pelec = new elecstate::ElecStatePW<T, Device>(this->pw_wfc,
                                                            &(this->chr),
                                                            &(this->kv),
                                                            &ucell,
                                                            &(GlobalC::ppcell),
                                                            this->pw_rhod,
                                                            this->pw_rho,
                                                            this->pw_big);
    }

    //! Inititlize the charge density.
    this->pelec->charge->allocate(GlobalV::NSPIN);

    //! set the cell volume variable in pelec
    this->pelec->omega = GlobalC::ucell.omega;

    // Initialize the potential.
    if (this->pelec->pot == nullptr)
    {
        this->pelec->pot = new elecstate::Potential(this->pw_rhod,
                                                    this->pw_rho,
                                                    &GlobalC::ucell,
                                                    &(GlobalC::ppcell.vloc),
                                                    &(this->sf),
                                                    &(this->pelec->f_en.etxc),
                                                    &(this->pelec->f_en.vtxc));
    }
    if (GlobalV::psi_initializer)
    {
        // update 20240320
        // use std::unique_ptr to manage the lifetime of psi_initializer under
        // restriction of C++11. Based on refactor of psi_initializer, the number
        // of member functions decrease significantly. 
        this->allocate_psi_init();
    }
    // temporary
    this->Init_GlobalC(inp, ucell);
    // Fix pelec->wg by ocp_kb
    if (GlobalV::ocp)
    {
        this->pelec->fixed_weights(GlobalV::ocp_kb);
    }
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::init_after_vc(Input& inp, UnitCell& ucell)
{
    ModuleBase::TITLE("ESolver_KS_PW", "init_after_vc");
    ModuleBase::timer::tick("ESolver_KS_PW", "init_after_vc");

    ESolver_KS<T, Device>::init_after_vc(inp, ucell);

    if (GlobalV::md_prec_level == 2)
    {
		this->pw_wfc->initgrids(
				ucell.lat0, 
				ucell.latvec, 
				this->pw_rho->nx, 
				this->pw_rho->ny, 
				this->pw_rho->nz);

		this->pw_wfc->initparameters(
				false, 
				inp.ecutwfc, 
				this->kv.nks, 
				this->kv.kvec_d.data());

#ifdef __MPI
		if (INPUT.pw_seed > 0)
		{
			MPI_Allreduce(MPI_IN_PLACE, &this->pw_wfc->ggecut, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		}
		// qianrui add 2021-8-13 to make different kpar parameters can get the same results
#endif

        this->pw_wfc->setuptransform();

        for (int ik = 0; ik < this->kv.nks; ++ik)
        {
            this->kv.ngk[ik] = this->pw_wfc->npwk[ik];
        }

        this->pw_wfc->collect_local_pw(inp.erf_ecut, inp.erf_height, inp.erf_sigma);

        delete this->phsol;
        this->phsol = new hsolver::HSolverPW<T, Device>(this->pw_wfc, &this->wf);

        delete this->pelec;
        this->pelec = new elecstate::ElecStatePW<T, Device>(this->pw_wfc,
                                                            &(this->chr),
                                                            (K_Vectors*)(&(this->kv)),
                                                            &ucell,
                                                            &(GlobalC::ppcell),
                                                            this->pw_rhod,
                                                            this->pw_rho,
                                                            this->pw_big);

        this->pelec->charge->allocate(GlobalV::NSPIN);

        //! setup cell volume
        this->pelec->omega = GlobalC::ucell.omega;

        delete this->pelec->pot;

        this->pelec->pot = new elecstate::Potential(this->pw_rhod,
                                                    this->pw_rho,
                                                    &GlobalC::ucell,
                                                    &(GlobalC::ppcell.vloc),
                                                    &(this->sf),
                                                    &(this->pelec->f_en.etxc),
                                                    &(this->pelec->f_en.vtxc));

        // temporary
        this->Init_GlobalC(inp, ucell);
    }
    else
    {
        GlobalC::ppcell.init_vnl(GlobalC::ucell, this->pw_rhod);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "NON-LOCAL POTENTIAL");

        this->pw_wfc->initgrids(GlobalC::ucell.lat0,
                                GlobalC::ucell.latvec,
                                this->pw_wfc->nx,
                                this->pw_wfc->ny,
                                this->pw_wfc->nz);

        this->pw_wfc->initparameters(false, INPUT.ecutwfc, this->kv.nks, this->kv.kvec_d.data());

        this->pw_wfc->collect_local_pw(inp.erf_ecut, inp.erf_height, inp.erf_sigma);
        
        if(GlobalV::psi_initializer) // new initialization method, used in KSDFT and LCAO_IN_PW calculation
        {
            // re-tabulate because GlobalV::DQ may change due to the change of atomic positions and cell parameters
            // for nao, we recalculate the overlap matrix between flz and jlq
            // for atomic, we recalculate the overlap matrix between pswfc and jlq
            // for psig is not read-only, its value will be overwritten in initialize_psi(), dont need delete and reallocate
            if((GlobalV::init_wfc.substr(0, 3) == "nao")
					||(GlobalV::init_wfc.substr(0, 6) == "atomic")) 
			{
				this->psi_init->tabulate(); 
			}
		}
        else // old initialization method, used in EXX calculation
        {
            this->wf.init_after_vc(this->kv.nks); // reallocate wanf2, the planewave expansion of lcao
            this->wf.init_at_1(&this->sf); // re-calculate tab_at, the overlap matrix between atomic pswfc and jlq
        }
    }

#ifdef USE_PAW
    if(GlobalV::use_paw)
    {
        GlobalC::paw_cell.set_libpaw_ecut(INPUT.ecutwfc/2.0,INPUT.ecutwfc/2.0); //in Hartree
        GlobalC::paw_cell.set_libpaw_fft(this->pw_wfc->nx,this->pw_wfc->ny,this->pw_wfc->nz,
                                         this->pw_wfc->nx,this->pw_wfc->ny,this->pw_wfc->nz,
                                         this->pw_wfc->startz,this->pw_wfc->numz);

#ifdef __MPI
		if(GlobalV::RANK_IN_POOL == 0) 
		{
			GlobalC::paw_cell.prepare_paw();
		}
#else
        GlobalC::paw_cell.prepare_paw();
#endif
        GlobalC::paw_cell.set_sij();

        std::vector<std::vector<double>> rhoijp;
        std::vector<std::vector<int>> rhoijselect;
        std::vector<int> nrhoijsel;
#ifdef __MPI
        if(GlobalV::RANK_IN_POOL == 0)
        {
            GlobalC::paw_cell.get_rhoijp(rhoijp, rhoijselect, nrhoijsel);

            for(int iat = 0; iat < GlobalC::ucell.nat; iat ++)
            {
                GlobalC::paw_cell.set_rhoij(iat,
						nrhoijsel[iat],
						rhoijselect[iat].size(),
						rhoijselect[iat].data(),
						rhoijp[iat].data());
            }  
        }
#else
        GlobalC::paw_cell.get_rhoijp(rhoijp, rhoijselect, nrhoijsel);

        for(int iat = 0; iat < GlobalC::ucell.nat; iat ++)
        {
			GlobalC::paw_cell.set_rhoij(iat,
					nrhoijsel[iat],
					rhoijselect[iat].size(),
					rhoijselect[iat].data(),
					rhoijp[iat].data());
		}
#endif
    }
#endif

    ModuleBase::timer::tick("ESolver_KS_PW", "init_after_vc");
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_scf(int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "before_scf");

    if (GlobalC::ucell.cell_parameter_updated)
    {
        this->init_after_vc(INPUT, GlobalC::ucell);
    }
    if (GlobalC::ucell.ionic_position_updated)
    {
        this->CE.update_all_dis(GlobalC::ucell);
        this->CE.extrapolate_charge(
#ifdef __MPI
            &(GlobalC::Pgrid),
#endif
            GlobalC::ucell,
            this->pelec->charge,
            &this->sf);
    }

    // init Hamilt, this should be allocated before each scf loop
    // Operators in HamiltPW should be reallocated once cell changed
    // delete Hamilt if not first scf
    if (this->p_hamilt != nullptr)
    {
        delete reinterpret_cast<hamilt::HamiltPW<T, Device>*>(this->p_hamilt);
        this->p_hamilt = nullptr;
    }

    // allocate HamiltPW
    if (this->p_hamilt == nullptr)
    {
		this->p_hamilt = new hamilt::HamiltPW<T, Device>(
				this->pelec->pot, 
				this->pw_wfc, 
				&this->kv);
    }

    //----------------------------------------------------------
    // about vdw, jiyy add vdwd3 and linpz add vdwd2
    //----------------------------------------------------------
    auto vdw_solver = vdw::make_vdw(GlobalC::ucell, INPUT);
    if (vdw_solver != nullptr)
    {
        this->pelec->f_en.evdw = vdw_solver->get_energy();
    }

    // calculate ewald energy
    if (!GlobalV::test_skip_ewald)
    {
		this->pelec->f_en.ewald_energy = H_Ewald_pw::compute_ewald(
				GlobalC::ucell, 
				this->pw_rhod, 
				this->sf.strucFac);
	}

    //! cal_ux should be called before init_scf because
    //! the direction of ux is used in noncoline_rho
    if(GlobalV::NSPIN == 4 && GlobalV::DOMAG) 
	{
		GlobalC::ucell.cal_ux();
	}

    //! calculate the total local pseudopotential in real space
    this->pelec->init_scf(istep, this->sf.strucFac);

    //! Symmetry_rho should behind init_scf, because charge should be initialized first.
    //! liuyu comment: Symmetry_rho should be located between init_rho and v_of_rho?
    Symmetry_rho srho;
    for (int is = 0; is < GlobalV::NSPIN; is++)
    {
		srho.begin(is, 
				*(this->pelec->charge), 
				this->pw_rhod, 
				GlobalC::Pgrid, 
				GlobalC::ucell.symm);
	}

    // liuyu move here 2023-10-09
    // D in uspp need vloc, thus behind init_scf()
    // calculate the effective coefficient matrix for non-local pseudopotential projectors
    ModuleBase::matrix veff = this->pelec->pot->get_effective_v();

    GlobalC::ppcell.cal_effective_D(veff, this->pw_rhod, GlobalC::ucell);

    // after init_rho (in pelec->init_scf), we have rho now.
    // before hamilt2density, we update Hk and initialize psi
    if(GlobalV::psi_initializer)
    {
        // before_scf function will be called everytime before scf. However, once atomic coordinates changed,
        // structure factor will change, therefore all atomwise properties will change. So we need to reinitialize
        // psi every time before scf. But for random wavefunction, we dont, because random wavefunction is not
        // related to atomic coordinates.
        // What the old strategy does is only to initialize for once...
        if(((GlobalV::init_wfc == "random")&&(istep == 0))
				||(GlobalV::init_wfc != "random")) 
		{
			this->initialize_psi();
		}
    }
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::others(const int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "others");
    ModuleBase::timer::tick("ESolver_KS_PW", "others");
    if (GlobalV::CALCULATION == "test_memory")
    {
        Cal_Test::test_memory(this->pw_rho,
                              this->pw_wfc,
                              this->p_chgmix->get_mixing_mode(),
                              this->p_chgmix->get_mixing_ndim());
        return;
    }

    if (GlobalV::CALCULATION == "gen_bessel")
    {
        // caoyu add 2020-11-24, mohan updat 2021-01-03
        Numerical_Descriptor nc;
        nc.output_descriptor(this->psi[0],
                             INPUT.bessel_descriptor_lmax,
                             INPUT.bessel_descriptor_rcut,
                             INPUT.bessel_descriptor_tolerence,
                             this->kv.nks);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "GENERATE DESCRIPTOR FOR DEEPKS");
        return;
    }

    // self consistent calculations for electronic ground state
    if (GlobalV::CALCULATION == "nscf")
    {
        this->nscf();
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::others", "CALCULATION type not supported");
    }

    ModuleBase::timer::tick("ESolver_KS_PW", "others");
    return;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_init(const int istep, const int iter)
{
    if (iter == 1)
    {
        this->p_chgmix->init_mixing();
        this->p_chgmix->mixing_restart_step = GlobalV::SCF_NMAX + 1;
    }
    // for mixing restart
    if (iter == this->p_chgmix->mixing_restart_step && GlobalV::MIXING_RESTART > 0.0)
    {
        this->p_chgmix->init_mixing();
    }
    // mohan move harris functional to here, 2012-06-05
    // use 'rho(in)' and 'v_h and v_xc'(in)
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband();

    //(2) save change density as previous charge,
    // prepared fox mixing.
    if (GlobalV::MY_STOGROUP == 0)
    {
        this->pelec->charge->save_rho_before_sum_band();
    }
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::allocate_psi_init()
{
    // under restriction of C++11, std::unique_ptr can not be allocate via std::make_unique
    // use new instead, but will cause asymmetric allocation and deallocation, in literal aspect
    ModuleBase::timer::tick("ESolver_KS_PW", "allocate_psi_init");
	if((GlobalV::init_wfc.substr(0, 6) == "atomic")&&(GlobalC::ucell.natomwfc == 0))
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
         new psi_initializer_random<T, Device>());
	}
	else if(GlobalV::init_wfc == "atomic") 
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
        new psi_initializer_atomic<T, Device>());
	}
    else if(GlobalV::init_wfc == "random") 
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
        new psi_initializer_random<T, Device>());
	}
    else if(GlobalV::init_wfc == "nao") 
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
        new psi_initializer_nao<T, Device>());
	}
	else if(GlobalV::init_wfc == "atomic+random") 
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
        new psi_initializer_atomic_random<T, Device>());
	}
	else if(GlobalV::init_wfc == "nao+random") 
	{
		this->psi_init = std::unique_ptr<psi_initializer<T, Device>>(
        new psi_initializer_nao_random<T, Device>());
	}
	else 
	{
		ModuleBase::WARNING_QUIT("ESolver_KS_PW::allocate_psi_init", 
        "for new psi initializer, init_wfc type not supported");
	}

    //! function polymorphism is moved from constructor to function initialize. 
    //! Two slightly different implementation are for MPI and serial case, respectively.
#ifdef __MPI
	this->psi_init->initialize(
			&this->sf, 
			this->pw_wfc, 
			&GlobalC::ucell, 
			&GlobalC::Pkpoints, 
			1, 
			&GlobalC::ppcell, 
			GlobalV::MY_RANK);
#else
	this->psi_init->initialize(
			&this->sf, 
			this->pw_wfc, 
			&GlobalC::ucell, 
			1, 
			&GlobalC::ppcell);
#endif

    // always new->initialize->tabulate->allocate->proj_ao_onkG
    this->psi_init->tabulate();
    ModuleBase::timer::tick("ESolver_KS_PW", "allocate_psi_init");

}



//! Although ESolver_KS_PW supports template, but in this function it has no relationship with
//! heterogeneous calculation, so all templates function are specialized to double
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::initialize_psi(void)
{
    ModuleBase::timer::tick("ESolver_KS_PW", "initialize_psi");
    if (GlobalV::psi_initializer)
    {
        for (int ik = 0; ik < this->pw_wfc->nks; ik++)
        {
            //! Fix the wavefunction to initialize at given kpoint
            this->psi->fix_k(ik);

            //! Update Hamiltonian from other kpoint to the given one
            this->p_hamilt->updateHk(ik);

            //! Project atomic orbitals on |k+G> planewave basis, where k is wavevector of kpoint
            //! and G is wavevector of the peroiodic part of the Bloch function
            this->psi_init->proj_ao_onkG(ik);

            //! psi_initializer manages memory of psig with shared pointer,
            //! its access to use is shared here via weak pointer
            //! therefore once the psi_initializer is destructed, psig will be destructed, too
            //! this way, we can avoid memory leak and undefined behavior
            std::weak_ptr<psi::Psi<T, Device>> psig = this->psi_init->share_psig();

			if(psig.expired()) 
			{
				ModuleBase::WARNING_QUIT("ESolver_KS_PW::initialize_psi", "psig lifetime is expired");
			}

            //! to use psig, we need to lock it to get a shared pointer version,
			//! then switch kpoint of psig to the given one
			auto psig_ = psig.lock(); 
			psig_->fix_k(ik);

            std::vector<Real> etatom(psig_->get_nbands(), 0.0);

            // then adjust dimension from psig to psi
            // either by matrix-multiplication or by copying-discarding
            if (this->psi_init->method() != "random")
            {
                // lcao_in_pw and pw share the same esolver. In the future, we will have different esolver
                if (
                      ((GlobalV::KS_SOLVER == "cg")||(GlobalV::KS_SOLVER == "lapack"))
                    &&(GlobalV::BASIS_TYPE == "pw") 
                    )
                {
                    // the following function is only run serially, to be improved
                    hsolver::DiagoIterAssist<T, Device>::diagH_subspace_init(
                        this->p_hamilt,
						psig_->get_pointer(), 
						psig_->get_nbands(), 
						psig_->get_nbasis(),
						*(this->kspw_psi), 
						etatom.data()
						);
                    continue;
                }
				else if ((GlobalV::KS_SOLVER == "lapack") 
						&& (GlobalV::BASIS_TYPE == "lcao_in_pw"))
				{
					if(ik == 0) 
					{
						GlobalV::ofs_running 
							<< " START WAVEFUNCTION: LCAO_IN_PW, psi initialization skipped " 
							<< std::endl;
					}
                    continue;
                }
                // else the case is davidson
            }
            else
            {
                if (GlobalV::KS_SOLVER == "cg")
                {
					hsolver::DiagoIterAssist<T, Device>::diagH_subspace(
							this->p_hamilt,
							*(psig_), 
							*(this->kspw_psi), 
							etatom.data()
							);
                    continue;
                }
                // else the case is davidson
            }

            // for the Davidson method, we just copy the wavefunction (partially)
            for (int iband = 0; iband < this->kspw_psi->get_nbands(); iband++)
            {
                for (int ibasis = 0; ibasis < this->kspw_psi->get_nbasis(); ibasis++)
                {
                    (*(this->kspw_psi))(iband, ibasis) = (*psig_)(iband, ibasis);
                }
            }
        }// end k-point loop

        this->psi_init->set_initialized(true);

    } // end GlobalV::psi_initializer 
    ModuleBase::timer::tick("ESolver_KS_PW", "initialize_psi");
}


// Temporary, it should be replaced by hsolver later.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2density(
		const int istep, 
		const int iter, 
		const double ethr)
{
    ModuleBase::timer::tick("ESolver_KS_PW", "hamilt2density");

    if (this->phsol != nullptr)
    {
        // reset energy
        this->pelec->f_en.eband = 0.0;
        this->pelec->f_en.demet = 0.0;
        // choose if psi should be diag in subspace
        // be careful that istep start from 0 and iter start from 1
        // if (iter == 1)
        hsolver::DiagoIterAssist<T, Device>::need_subspace = ((istep == 0 || istep == 1) && iter == 1)? false : true;
        hsolver::DiagoIterAssist<T, Device>::SCF_ITER = iter;
        hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR = ethr;
        hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX = GlobalV::PW_DIAG_NMAX;

        // after init_rho (in pelec->init_scf), we have rho now.
        // before hamilt2density, we update Hk and initialize psi
        if(GlobalV::psi_initializer)
        {
            // before_scf function will be called everytime before scf. However, once atomic coordinates changed,
            // structure factor will change, therefore all atomwise properties will change. So we need to reinitialize
            // psi every time before scf. But for random wavefunction, we dont, because random wavefunction is not
            // related to atomic coordinates.

            // What the old strategy does is only to initialize for once... we also initialize only once here because
            // this can save a lot of time. But if cell and ion change significantly, re-initialization psi will be
			// more efficient. Or an extrapolation strategy can be used.
			if((istep == 0)&&(iter == 1)
					&&!(this->psi_init->initialized())) 
			{
				this->initialize_psi();
			}
        }
        if(GlobalV::BASIS_TYPE != "lcao_in_pw")
        {
            // from HSolverPW
            this->phsol->solve(this->p_hamilt,      // hamilt::Hamilt<T, Device>* pHamilt,
                               this->kspw_psi[0],   // psi::Psi<T, Device>& psi,
                               this->pelec,         // elecstate::ElecState<T, Device>* pelec,
                               GlobalV::KS_SOLVER); // const std::string method_in,
        }
        else
        {
            // It is not a good choice to overload another solve function here, this will spoil the concept of 
            // multiple inheritance and polymorphism. But for now, we just do it in this way.
            // In the future, there will be a series of class ESolver_KS_LCAO_PW, HSolver_LCAO_PW and so on.
            std::weak_ptr<psi::Psi<T, Device>> psig = this->psi_init->share_psig();

			if(psig.expired()) 
			{
				ModuleBase::WARNING_QUIT("ESolver_KS_PW::hamilt2density", "psig lifetime is expired");
			}

			// from HSolverPW
            this->phsol->solve(this->p_hamilt,          // hamilt::Hamilt<T, Device>* pHamilt,
                               this->kspw_psi[0],       // psi::Psi<T, Device>& psi,
                               this->pelec,             // elecstate::ElecState<T, Device>* pelec,
                               psig.lock().get()[0]);   // psi::Psi<T, Device>& transform,
        }
        if (GlobalV::out_bandgap)
        {
			if (!GlobalV::TWO_EFERMI) 
			{
				this->pelec->cal_bandgap();
			}
			else 
			{
				this->pelec->cal_bandgap_updw();
			}
		}
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW", "HSolver has not been initialed!");
    }
    // add exx
#ifdef __LCAO
#ifdef __EXX
    this->pelec->set_exx(GlobalC::exx_lip.get_exx_energy()); // Peize Lin add 2019-03-09
#endif
#endif

    // calculate the delta_harris energy
    // according to new charge density.
    // mohan add 2009-01-23
    this->pelec->cal_energies(1);

    Symmetry_rho srho;
    for (int is = 0; is < GlobalV::NSPIN; is++)
	{
		srho.begin(is, 
				*(this->pelec->charge), 
				this->pw_rhod, 
				GlobalC::Pgrid, 
				GlobalC::ucell.symm);
	}

    // compute magnetization, only for LSDA(spin==2)
    GlobalC::ucell.magnet.compute_magnetization(this->pelec->charge->nrxx,
                                                this->pelec->charge->nxyz,
                                                this->pelec->charge->rho,
                                                this->pelec->nelec_spin.data());

    // deband is calculated from "output" charge density calculated
    // in sum_band
    // need 'rho(out)' and 'vr (v_h(in) and v_xc(in))'
    this->pelec->f_en.deband = this->pelec->cal_delta_eband();

    ModuleBase::timer::tick("ESolver_KS_PW", "hamilt2density");
}


// Temporary, it should be rewritten with Hamilt class.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::update_pot(const int istep, const int iter)
{
    if (!this->conv_elec)
    {
		if (GlobalV::NSPIN == 4)
		{
			GlobalC::ucell.cal_ux();
		}
		this->pelec->pot->update_from_charge(this->pelec->charge, &GlobalC::ucell);
        this->pelec->f_en.descf = this->pelec->cal_delta_escf();
    }
    else
    {
        this->pelec->cal_converged();
    }
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_finish(const int iter)
{
    // liuyu 2023-10-24
    // D in uspp need vloc, thus needs update when veff updated
    // calculate the effective coefficient matrix for non-local pseudopotential projectors
    if (GlobalV::use_uspp)
    {
        ModuleBase::matrix veff = this->pelec->pot->get_effective_v();
        GlobalC::ppcell.cal_effective_D(veff, this->pw_rhod, GlobalC::ucell);
    }

    this->pelec->cal_energies(2);

    // We output it for restarting the scf.
    bool print = false;
    if (this->out_freq_elec && iter % this->out_freq_elec == 0)
    {
        print = true;
    }

    if (print)
    {
        if (GlobalV::out_chg > 0)
        {
            for (int is = 0; is < GlobalV::NSPIN; is++)
            {
                this->create_Output_Rho(is, iter, "tmp_").write();
                if (XC_Functional::get_func_type() == 3 || XC_Functional::get_func_type() == 5)
                {
                    this->create_Output_Kin(is, iter, "tmp_").write();
                }
            }
        }
        // output wavefunctions
        if (this->wf.out_wfc_pw == 1 || this->wf.out_wfc_pw == 2)
        {
            std::stringstream ssw;
            ssw << GlobalV::global_out_dir << "WAVEFUNC";
            // mohan update 2011-02-21
            // qianrui update 2020-10-17
            ModuleIO::write_wfc_pw(ssw.str(), this->psi[0], this->kv, this->pw_wfc);
            // ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running,"write wave functions into file WAVEFUNC.dat");
        }
    }
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_scf(const int istep)
{
    this->create_Output_Potential(istep).write();

    // save charge difference into files for charge extrapolation
    if (GlobalV::CALCULATION != "scf")
    {
        this->CE.save_files(istep,
                            GlobalC::ucell,
#ifdef __MPI
                            this->pw_big,
#endif
                            this->pelec->charge,
                            &this->sf);
    }

    if (GlobalV::out_chg)
    {
        for (int is = 0; is < GlobalV::NSPIN; is++)
        {
            this->create_Output_Rho(is, istep).write();
            if (XC_Functional::get_func_type() == 3 || XC_Functional::get_func_type() == 5)
            {
                this->create_Output_Kin(is, istep).write();
            }
        }
    }

	if (this->wf.out_wfc_pw == 1 
			|| this->wf.out_wfc_pw == 2)
	{
        std::stringstream ssw;
        ssw << GlobalV::global_out_dir << "WAVEFUNC";
        ModuleIO::write_wfc_pw(ssw.str(), this->psi[0], this->kv, this->pw_wfc);
    }

    ModuleIO::output_convergence_after_scf(this->conv_elec, this->pelec->f_en.etot);

    ModuleIO::output_efermi(this->conv_elec, this->pelec->eferm.ef); 

    if (GlobalV::OUT_LEVEL != "m")
    {
        this->pelec->print_eigenvalue(GlobalV::ofs_running);
    }

    if (this->device == base_device::GpuDevice)
    {
        castmem_2d_d2h_op()(this->psi[0].get_device(),
                            this->kspw_psi[0].get_device(),
                            this->psi[0].get_pointer() - this->psi[0].get_psi_bias(),
                            this->kspw_psi[0].get_pointer() - this->kspw_psi[0].get_psi_bias(),
                            this->psi[0].size());
    }

    // Get bands_to_print through public function of INPUT (returns a const pointer to string)
    std::string bands_to_print = *INPUT.get_bands_to_print();
    if(!bands_to_print.empty())
    {
        std::vector<double> out_band_kb;
        Input_Conv::parse_expression(bands_to_print, out_band_kb);

        // bands_picked is a vector of 0s and 1s, where 1 means the band is picked to output
        std::vector<int> bands_picked;
        bands_picked.resize(this->kspw_psi->get_nbands());
        ModuleBase::GlobalFunc::ZEROS(bands_picked.data(), this->kspw_psi->get_nbands());

        // Check if length of out_band_kb is valid
        if (static_cast<int>(out_band_kb.size()) > this->kspw_psi->get_nbands())
        {
            ModuleBase::WARNING_QUIT(
                "ESolver_KS_PW::after_scf",
                "The number of bands specified by `bands_to_print` in the INPUT file exceeds `nbands`!");
        }

        // Check if all elements in bands_picked are 0 or 1
        for (int value: out_band_kb)
        {
            if (value != 0 && value != 1)
            {
                ModuleBase::WARNING_QUIT(
                    "ESolver_KS_PW::after_scf",
                    "The elements of `bands_to_print` must be either 0 or 1. Invalid values found!");
            }
        }

        // Fill bands_picked with values from out_band_kb, converting to int
        // Remaining bands are already set to 0
        int length = std::min(static_cast<int>(out_band_kb.size()), this->kspw_psi->get_nbands());
        for (int i = 0; i < length; ++i)
        {
            // out_band_kb rely on function parse_expression from input_conv.cpp
            // Initially designed for ocp_set, which can be double
            bands_picked[i] = static_cast<int>(out_band_kb[i]);
        }

        std::complex<double>* wfcr = new std::complex<double>[this->pw_rho->nxyz];
        double* rho_band = new double[this->pw_rho->nxyz];

        for (int ib = 0; ib < this->kspw_psi->get_nbands(); ++ib)
        {
            // Skip the loop iteration if bands_picked[ib] is 0
            if (!bands_picked[ib])
            {
                continue;
            } 

            for (int i = 0; i < this->pw_rho->nxyz; i++)
            {
                // Initialize rho_band to zero for each band
                rho_band[i] = 0.0;
            }

            for (int ik = 0; ik < this->kv.nks; ik++)
            {
                this->psi->fix_k(ik);
                this->pw_wfc->recip_to_real(this->ctx, &psi[0](ib, 0), wfcr, ik);

                double w1 = static_cast<double>(this->kv.wk[ik] / GlobalC::ucell.omega);

                for (int i = 0; i < this->pw_rho->nxyz; i++)
                {
                    rho_band[i] += std::norm(wfcr[i]) * w1;
                }
            }

            std::stringstream ssc;
            ssc << GlobalV::global_out_dir << "band" << ib + 1 << ".cube"; // band index starts from 1

            ModuleIO::write_rho(
#ifdef __MPI
                this->pw_big->bz,
                this->pw_big->nbz,
                this->pw_big->nplane,
                this->pw_big->startz_current,
#endif
                rho_band,
                0,
                GlobalV::NSPIN,
                0,
                ssc.str(),
                this->pw_rho->nx,
                this->pw_rho->ny,
                this->pw_rho->nz,
                0.0,
                &(GlobalC::ucell),
                11);
        }
        delete[] wfcr;
        delete[] rho_band;
    }
}


template <typename T, typename Device>
double ESolver_KS_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_force(ModuleBase::matrix& force)
{
    Forces<double, Device> ff(GlobalC::ucell.nat);
	if (this->__kspw_psi != nullptr)
	{
		this->__kspw_psi = nullptr;
	}

	if (this->__kspw_psi == nullptr)
    {
        this->__kspw_psi = GlobalV::precision_flag == "single"
                               ? new psi::Psi<std::complex<double>, Device>(this->kspw_psi[0])
                               : reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->kspw_psi);
    }

    //! Calculate forces
	ff.cal_force(force, 
			*this->pelec, 
			this->pw_rhod, 
			&GlobalC::ucell.symm, 
			&this->sf, 
			&this->kv, 
			this->pw_wfc, 
			this->__kspw_psi);
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_stress(ModuleBase::matrix& stress)
{
    Stress_PW<double, Device> ss(this->pelec);
    if (this->__kspw_psi != nullptr)
    {
        this->__kspw_psi = nullptr;
    }

    if (this->__kspw_psi == nullptr)
    {
		this->__kspw_psi = GlobalV::precision_flag == "single"
			? new psi::Psi<std::complex<double>, Device>(this->kspw_psi[0])
			: reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->kspw_psi);
	}
	ss.cal_stress(stress,
			GlobalC::ucell,
			this->pw_rhod,
			&GlobalC::ucell.symm,
			&this->sf,
			&this->kv,
			this->pw_wfc,
			this->psi,
			this->__kspw_psi);

	// external stress
	double unit_transform = 0.0;
	unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
	double external_stress[3] = {GlobalV::PRESS1, GlobalV::PRESS2, GlobalV::PRESS3};
    for (int i = 0; i < 3; i++)
    {
        stress(i, i) -= external_stress[i] / unit_transform;
    }
    GlobalV::PRESSURE = (stress(0, 0) + stress(1, 1) + stress(2, 2)) / 3;
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_all_runners(void)
{

    GlobalV::ofs_running << "\n\n --------------------------------------------" << std::endl;
    GlobalV::ofs_running << std::setprecision(16);
    GlobalV::ofs_running << " !FINAL_ETOT_IS " << this->pelec->f_en.etot * ModuleBase::Ry_to_eV << " eV" << std::endl;
    GlobalV::ofs_running << " --------------------------------------------\n\n" << std::endl;

    if (INPUT.out_dos != 0 || INPUT.out_band[0] != 0)
    {
        GlobalV::ofs_running << "\n\n\n\n";
        GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        GlobalV::ofs_running << " |                                                                    |" << std::endl;
        GlobalV::ofs_running << " | Post-processing of data:                                           |" << std::endl;
        GlobalV::ofs_running << " | DOS (density of states) and bands will be output here.             |" << std::endl;
        GlobalV::ofs_running << " | If atomic orbitals are used, Mulliken charge analysis can be done. |" << std::endl;
        GlobalV::ofs_running << " | Also the .bxsf file containing fermi surface information can be    |" << std::endl;
		GlobalV::ofs_running << " | done here.                                                         |" << std::endl;
		GlobalV::ofs_running << " |                                                                    |" << std::endl;
		GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
		GlobalV::ofs_running << "\n\n\n\n";
	}

	int nspin0 = 1;
	if (GlobalV::NSPIN == 2)
	{
		nspin0 = 2;
	}
	//! print occupation in istate.info
	ModuleIO::write_istate_info(this->pelec->ekb, this->pelec->wg, this->kv, &(GlobalC::Pkpoints));

	//! compute density of states
    if (INPUT.out_dos)
    {
        ModuleIO::write_dos_pw(this->pelec->ekb,
                               this->pelec->wg,
                               this->kv,
                               INPUT.dos_edelta_ev,
                               INPUT.dos_scale,
                               INPUT.dos_sigma);

        if (nspin0 == 1)
        {
            GlobalV::ofs_running << " Fermi energy is " << this->pelec->eferm.ef << " Rydberg" << std::endl;
        }
        else if (nspin0 == 2)
		{
			GlobalV::ofs_running << " Fermi energy (spin = 1) is " 
				<< this->pelec->eferm.ef_up 
				<< " Rydberg"
				<< std::endl;
			GlobalV::ofs_running << " Fermi energy (spin = 2) is " 
				<< this->pelec->eferm.ef_dw 
				<< " Rydberg"
				<< std::endl;
		}
    }


    if (INPUT.out_band[0]) // pengfei 2014-10-13
    {
        for (int is = 0; is < nspin0; is++)
        {
            std::stringstream ss2;
            ss2 << GlobalV::global_out_dir << "BANDS_" << is + 1 << ".dat";
            GlobalV::ofs_running << "\n Output bands in file: " << ss2.str() << std::endl;
            ModuleIO::nscf_band(is,
                                ss2.str(),
                                GlobalV::NBANDS,
                                0.0,
                                INPUT.out_band[1],
                                this->pelec->ekb,
                                this->kv,
                                &(GlobalC::Pkpoints));
        }
    }


    if (GlobalV::BASIS_TYPE == "pw" && winput::out_spillage) // xiaohui add 2013-09-01
    {
        // std::cout << "\n Output Spillage Information : " << std::endl;
        //  calculate spillage value.
#ifdef __LCAO
// We are not goint to support lcao_in_paw until
// the obsolete GlobalC::hm is replaced by the
// refactored moeules (psi, hamilt, etc.)
/*
            if ( winput::out_spillage == 3)
            {
                GlobalV::BASIS_TYPE="pw";
                std::cout << " NLOCAL = " << GlobalV::NLOCAL << std::endl;

                for (int ik=0; ik<this->kv.nks; ik++)
                {
                    this->wf.wanf2[ik].create(GlobalV::NLOCAL, this->wf.npwx);
                    if(GlobalV::BASIS_TYPE=="pw")
                    {
                        std::cout << " ik=" << ik + 1 << std::endl;

                        GlobalV::BASIS_TYPE="lcao_in_pw";
                        this->wf.LCAO_in_pw_k(ik, this->wf.wanf2[ik]);
                        GlobalV::BASIS_TYPE="pw";
                    }
                }

                //Spillage sp;
                //sp.get_both(GlobalV::NBANDS, GlobalV::NLOCAL, this->wf.wanf2, this->wf.evc);
            }
*/
#endif

        // ! Print out overlap before spillage optimization to generate atomic orbitals
        if (winput::out_spillage <= 2)
        {
            if(INPUT.bessel_nao_rcuts.size() == 1)
            {
                Numerical_Basis numerical_basis;
                numerical_basis.output_overlap(this->psi[0], this->sf, this->kv, this->pw_wfc, GlobalC::ucell);
            }
            else
            {
                for(int i = 0; i < INPUT.bessel_nao_rcuts.size(); i++)
                {
					if(GlobalV::MY_RANK == 0) 
					{
						std::cout << "update value: bessel_nao_rcut <- " 
							<< std::fixed 
							<< INPUT.bessel_nao_rcuts[i] 
							<< " a.u." 
							<< std::endl;
					}
					INPUT.bessel_nao_rcut = INPUT.bessel_nao_rcuts[i];
                    /*
                        SEVERE BUG
                        the memory management of numerical_basis class is NOT SAFE, 
                        data cleaning before overwriting is absent.
                        instance created from present implementation of numerical_basis 
                        SHOULD NOT BE USED FOR MORE THAN ONE TIME.
                        will cause data unexpected overwriting, file truncation and data loss. 
                        Will be refactored in the future.
                    */
                    Numerical_Basis numerical_basis;
                    numerical_basis.output_overlap(this->psi[0], this->sf, this->kv, this->pw_wfc, GlobalC::ucell);
					std::string old_fname_header = winput::spillage_outdir 
						+ "/" 
						+ "orb_matrix.";
					std::string new_fname_header = winput::spillage_outdir 
						+ "/" 
						+ "orb_matrix_rcut" 
						+ std::to_string(int(INPUT.bessel_nao_rcut)) 
						+ "deriv";

                    for(int derivative_order = 0; derivative_order <= 1; derivative_order++)
                    {
                        // rename generated files
                        std::string old_fname = old_fname_header + std::to_string(derivative_order) + ".dat";
                        std::string new_fname = new_fname_header + std::to_string(derivative_order) + ".dat";
                        std::rename(old_fname.c_str(), new_fname.c_str());
                    }
                }
            }
            ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "BASIS OVERLAP (Q and S) GENERATION.");
        }
    }

    //! Print out wave functions in real space
    if (this->wf.out_wfc_r == 1) // Peize Lin add 2021.11.21
    {
		ModuleIO::write_psi_r_1(
				this->psi[0], 
				this->pw_wfc, 
				"wfc_realspace", 
				true, 
				this->kv);
    }

    //! Use Kubo-Greenwood method to compute conductivities
    if (INPUT.cal_cond)
    {
		this->KG(
				INPUT.cond_smear, 
				INPUT.cond_fwhm, 
				INPUT.cond_wcut, 
				INPUT.cond_dw, 
				INPUT.cond_dt, 
				this->pelec->wg);
	}
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2estates(const double ethr)
{
    if (this->phsol != nullptr)
    {
        hsolver::DiagoIterAssist<T, Device>::need_subspace = false;
        hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR = ethr;
        this->phsol->solve(this->p_hamilt, this->kspw_psi[0], this->pelec, GlobalV::KS_SOLVER, true);
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW", "HSolver has not been initialed!");
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::nscf(void)
{
    ModuleBase::TITLE("ESolver_KS_PW", "nscf");
    ModuleBase::timer::tick("ESolver_KS_PW", "nscf");

    // mohan add istep_tmp 2024-03-31
    const int istep_tmp = 0;
    this->before_scf(istep_tmp);

    //! Setup the parameters for diagonalization
    double diag_ethr = GlobalV::PW_DIAG_THR;
    if (diag_ethr - 1e-2 > -1e-5)
    {
        diag_ethr = std::max(1e-13, 0.1 * std::min(1e-2, GlobalV::SCF_THR / GlobalV::nelec));
    }
    GlobalV::ofs_running << " PW_DIAG_THR  = " << diag_ethr << std::endl;

    //! Diagonalize Hamiltonian
    this->hamilt2estates(diag_ethr);

    //! Calculate weights/Fermi energies
    this->pelec->calculate_weights();


    GlobalV::ofs_running << "\n End of Band Structure Calculation \n" << std::endl;

    //! Print out band energies and weights
    for (int ik = 0; ik < this->kv.nks; ik++)
    {
        if (GlobalV::NSPIN == 2)
        {
            if (ik == 0)
			{
				GlobalV::ofs_running << " spin up :" << std::endl;
			}
			if (ik == (this->kv.nks / 2))
			{
				GlobalV::ofs_running << " spin down :" << std::endl;
			}
        }

		GlobalV::ofs_running << " k-points" << ik + 1 
			<< "(" << this->kv.nkstot 
			<< "): " << this->kv.kvec_c[ik].x 
			<< " "
			<< this->kv.kvec_c[ik].y 
			<< " " << this->kv.kvec_c[ik].z << std::endl;

        for (int ib = 0; ib < GlobalV::NBANDS; ib++)
        {
			GlobalV::ofs_running << " spin" 
				<< this->kv.isk[ik] + 1 
				<< "_final_band " 
				<< ib + 1 << " "
				<< this->pelec->ekb(ik, ib) * ModuleBase::Ry_to_eV << " "
				<< this->pelec->wg(ik, ib) * this->kv.nks 
				<< std::endl;
		}
        GlobalV::ofs_running << std::endl;
    }

    //! Print out band gaps
    if (GlobalV::out_bandgap)
    {
        if (!GlobalV::TWO_EFERMI)
        {
            this->pelec->cal_bandgap();
			GlobalV::ofs_running << " E_bandgap " 
				<< this->pelec->bandgap * ModuleBase::Ry_to_eV 
				<< " eV" << std::endl;
		}
        else
        {
			this->pelec->cal_bandgap_updw();
			GlobalV::ofs_running << " E_bandgap_up " 
				<< this->pelec->bandgap_up * ModuleBase::Ry_to_eV 
				<< " eV"
				<< std::endl;
			GlobalV::ofs_running << " E_bandgap_dw " 
				<< this->pelec->bandgap_dw * ModuleBase::Ry_to_eV 
				<< " eV"
				<< std::endl;
		}
    }

    //! Calculate Wannier functions
    // add by jingan in 2018.11.7
    if (INPUT.towannier90)
    {
        toWannier90_PW myWannier(
            INPUT.out_wannier_mmn,
            INPUT.out_wannier_amn,
            INPUT.out_wannier_unk, 
            INPUT.out_wannier_eig,
            INPUT.out_wannier_wvfn_formatted, 
            INPUT.nnkpfile,
            INPUT.wannier_spin
        );

		myWannier.calculate(
				this->pelec->ekb, 
				this->pw_wfc, 
				this->pw_big, 
				this->kv, 
				this->psi);
	}


    //! Calculate Berry phase polarization 
    if (berryphase::berry_phase_flag 
        && ModuleSymmetry::Symmetry::symm_flag != 1)
    {
        berryphase bp;
		bp.Macroscopic_polarization(
				this->pw_wfc->npwk_max, 
				this->psi, 
				this->pw_rho, 
				this->pw_wfc, 
				this->kv);
	}

    ModuleBase::timer::tick("ESolver_KS_PW", "nscf");
    return;
}

template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver
