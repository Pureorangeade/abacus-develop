#include "langevin.h"

#include "md_func.h"
#include "module_base/parallel_common.h"
#include "module_base/timer.h"

Langevin::Langevin(MD_para& MD_para_in, UnitCell& unit_in) : MD_base(MD_para_in, unit_in)
{
    /// convert to a.u. unit
    assert(ModuleBase::AU_to_FS!=0.0);

    mdp.md_damp /= ModuleBase::AU_to_FS;

    assert(ucell.nat>0);

    total_force = new ModuleBase::Vector3<double>[ucell.nat];
}

Langevin::~Langevin()
{
    delete[] total_force;
}


void Langevin::setup(ModuleESolver::ESolver* p_esolver, const std::string& global_readin_dir)
{
    ModuleBase::TITLE("Langevin", "setup");
    ModuleBase::timer::tick("Langevin", "setup");

    MD_base::setup(p_esolver, global_readin_dir);

    post_force();

    ModuleBase::timer::tick("Langevin", "setup");
    return;
}


void Langevin::first_half(std::ofstream& ofs)
{
    ModuleBase::TITLE("Langevin", "first_half");
    ModuleBase::timer::tick("Langevin", "first_half");

    MD_base::update_vel(total_force);
    MD_base::update_pos();

    ModuleBase::timer::tick("Langevin", "first_half");
    return;
}


void Langevin::second_half(void)
{
    ModuleBase::TITLE("Langevin", "second_half");
    ModuleBase::timer::tick("Langevin", "second_half");

    post_force();
    MD_base::update_vel(total_force);

    ModuleBase::timer::tick("Langevin", "second_half");
    return;
}


void Langevin::print_md(std::ofstream& ofs, const bool& cal_stress)
{
    MD_base::print_md(ofs, cal_stress);
    return;
}


void Langevin::write_restart(const std::string& global_out_dir)
{
    MD_base::write_restart(global_out_dir);
    return;
}


void Langevin::restart(const std::string& global_readin_dir)
{
    MD_base::restart(global_readin_dir);
    return;
}


void Langevin::post_force(void)
{
    double t_target = MD_func::target_temp(step_ + step_rst_, mdp.md_nstep, mdp.md_tfirst, mdp.md_tlast);
    ModuleBase::Vector3<double> fictitious_force;
#ifdef __MPI
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int each_ucell_nat = ucell.nat / size;
    int ucell_nat_begin = (each_ucell_nat) * mdp.my_rank;
    int ucell_nat_end = ucell_nat_begin + each_ucell_nat;
    if(mdp.my_rank == size - 1)
    {
        ucell_nat_end = ucell.nat;
    }
    for (int i = ucell_nat_begin; i < ucell_nat_end; ++i)
    {
        fictitious_force = -allmass[i] * vel[i] / mdp.md_damp;
        for (int j = 0; j < 3; ++j)
        {
            fictitious_force[j] += sqrt(24.0 * t_target * allmass[i] / mdp.md_damp / mdp.md_dt)
                                       * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
        }
        total_force[i] = force[i] + fictitious_force;
    }
for(int i = 0; i < size; i++){
    int each_ucell_nat = ucell.nat / size;
    int ucell_nat_begin = (each_ucell_nat) * i;
    int ucell_nat_end = ucell_nat_begin + each_ucell_nat;
    if(i == size - 1)
    {
        ucell_nat_end = ucell.nat;
    }
    MPI_Bcast(total_force + ucell_nat_begin, (ucell_nat_end - ucell_nat_begin) * 3, MPI_DOUBLE, i, MPI_COMM_WORLD);
}
#endif
}
