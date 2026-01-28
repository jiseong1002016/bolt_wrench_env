#include "Environment.hpp"
#include "rsc/sweep_test.hpp"

namespace raisim {

/// Run the FT vs PD comparison sweep from Python.
/// Returns rows of: [dt, cutoff_hz, num_fail, num_samples].
std::vector<std::array<double, 4>> ENVIRONMENT::runFtPdSweep(
    double dt_min,
    double dt_max,
    double dt_step,
    double cutoff_min,
    double cutoff_max,
    double cutoff_step,
    double sample_time,
    int num_samples,
    bool random_dt_step,
    const std::string& urdf_type) const {
  std::vector<std::array<double, 4>> results;

  if (dt_step <= 0.0 || cutoff_step <= 0.0 || num_samples <= 0 || sample_time <= 0.0) {
    return results;
  }

  raisim::World sweep_world;
  const std::string urdf_prefix =
      resourceDir_ + "/RAIPAL-URDF/urdf/raipal" + urdf_type;

  auto* raipal_FT = sweep_world.addArticulatedSystem(urdf_prefix + "_R.urdf");
  auto* raipal_PD = sweep_world.addArticulatedSystem(urdf_prefix + "_L.urdf");
  raipal_FT->setName("raipal_FT");
  raipal_PD->setName("raipal_PD");
  raipal_FT->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
  raipal_PD->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

  const int gcDim = raipal_FT->getGeneralizedCoordinateDim();
  const int gvDim = raipal_FT->getDOF();

  Eigen::VectorXd gc = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd gv = Eigen::VectorXd::Zero(gvDim);
  Eigen::VectorXd gcRaw = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd gvRaw = Eigen::VectorXd::Zero(gvDim);
  Eigen::VectorXd gc_init = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd gv_init = Eigen::VectorXd::Zero(gvDim);
  Eigen::VectorXd pTarget = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd dTarget = Eigen::VectorXd::Zero(gvDim);

  Eigen::VectorXd jointPgain = Eigen::VectorXd::Ones(gvDim) * 200.0;
  Eigen::VectorXd jointDgain = Eigen::VectorXd::Ones(gvDim) * 20.0;
  if (gvDim > 5) {
    jointPgain[4] = 0.0;
    jointPgain[5] = 0.0;
    jointDgain[4] = 0.0;
    jointDgain[5] = 0.0;
  }

  raipal_PD->setPdGains(jointPgain, jointDgain);

  std::mt19937 gen(
      static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
  std::uniform_real_distribution<double> uniDist(0.0, 1.0);

  const auto joint_limits = raipal_FT->getJointLimits();

  Eigen::VectorXd gc_FT = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd gv_FT = Eigen::VectorXd::Zero(gvDim);
  Eigen::VectorXd gc_PD = Eigen::VectorXd::Zero(gcDim);
  Eigen::VectorXd gv_PD = Eigen::VectorXd::Zero(gvDim);

  const double eps = 1e-10;
  for (double dt_range_max = dt_min; dt_range_max <= dt_max + eps; dt_range_max += dt_step) {
    const double dt_range_min = dt_min;
    for (double cutoff_freq = cutoff_min; cutoff_freq <= cutoff_max + eps;
         cutoff_freq += cutoff_step) {
      int numFail = 0;

      for (int sampleIdx = 0; sampleIdx < num_samples; ++sampleIdx) {
        sweep_test_detail::sampleTarget(pTarget, joint_limits, gen, uniDist);

        raipal_FT->setState(gc_init, gv_init);
        raipal_PD->setState(gc_init, gv_init);
        raipal_FT->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim));
        raipal_PD->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim));
        raipal_PD->setPdTarget(pTarget, dTarget);

        double dt = dt_range_max;
        sweep_world.setTimeStep(dt);
        sweep_world.setWorldTime(0.0);

        bool fail = false;
        bool declared = false;
        double elapsed_time = 0.0;

        while (sweep_world.getWorldTime() < sample_time) {
          raipal_FT->getState(gcRaw, gvRaw);
          const double alpha =
              (2.0 * M_PI * cutoff_freq * dt) / (2.0 * M_PI * cutoff_freq * dt + 1.0);
          gc = alpha * gcRaw + (1.0 - alpha) * gc;
          gv = alpha * gvRaw + (1.0 - alpha) * gv;

          Eigen::VectorXd tau = Eigen::VectorXd::Zero(gvDim);
          for (int i = 0; i < gcDim; ++i) {
            const double p_err = pTarget[i] - gc[i];
            const double d_err = dTarget[i] - gv[i];
            tau[i] = jointPgain[i] * p_err + jointDgain[i] * d_err;
          }
          raipal_FT->setGeneralizedForce(tau);

          sweep_world.integrate();

          raipal_FT->getState(gc_FT, gv_FT);
          raipal_PD->getState(gc_PD, gv_PD);

          const bool gc_diff = (gc_FT - gc_PD).cwiseAbs().maxCoeff() > 1e-3;
          const bool gv_diff = (gv_FT - gv_PD).cwiseAbs().maxCoeff() > 1e-2;
          fail = gc_diff || gv_diff;

          elapsed_time += dt;
          if (elapsed_time < 1.5) {
            fail = false;
          }

          if (fail && !declared) {
            ++numFail;
            declared = true;
            break;
          }

          if (random_dt_step) {
            dt = dt_range_min + (dt_range_max - dt_range_min) * uniDist(gen);
            sweep_world.setTimeStep(dt);
          }
        }
      }

      results.push_back({dt_range_max, cutoff_freq, static_cast<double>(numFail),
                         static_cast<double>(num_samples)});

      if (numFail == num_samples) {
        break;
      }
    }
  }

  return results;
}

}  // namespace raisim
