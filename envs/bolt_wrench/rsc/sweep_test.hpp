#pragma once

#include <random>
#include <vector>
#include <array>

#include <Eigen/Core>

namespace raisim {
namespace sweep_test_detail {

/// Crossed 4-bar linkage forward kinematics calculation.
/// Calculates indices 4/5 of gc based on index 3.
inline void cfbFK(Eigen::VectorXd& gc) {
  static const double coeffGC5[17] = {
      7.487348861572753,   -88.17268027058002, 470.7676589474978,
      -1507.949571146181,  3232.924175366478,  -4903.587201465159,
      5428.248858581333,   -4464.681332539945, 2751.745273804874,
      -1265.589676288956,  418.4687974259759,  -86.36058832217348,
      6.540591054554072,   -1.820824957906633, 2.514881046035373,
      1.022698355939166,   0.000002629580961635315};
  static const double coeffGC4[17] = {
      -4.599352352699746,  54.74231689702444,  -295.2311950384566,
      953.4731854791298,   -2052.475077842654, 3099.83081925634,
      -3363.608943070999,  2636.542385991094,  -1477.728449862623,
      579.0562233124353,   -157.9024440428828, 35.05094543077983,
      -7.773109367008759,  -0.656408504334005, 0.798469056970581,
      2.098468226080274,   -0.000001063969759856979};

  gc[5] = 0.0;
  gc[4] = 0.0;
  double gc3_pow = 1.0;
  for (int i = 16; i >= 0; --i) {
    gc[5] += coeffGC5[i] * gc3_pow;
    gc[4] += coeffGC4[i] * gc3_pow;
    gc3_pow *= gc[3];
  }
}

inline void sampleTarget(Eigen::VectorXd& pTarget,
                         const std::vector<raisim::Vec<2UL>>& jointLimits,
                         std::mt19937& gen,
                         std::uniform_real_distribution<double>& uniDist) {
  const int limit_dim = static_cast<int>(jointLimits.size());
  for (int i = 0; i < limit_dim; ++i) {
    const double lo = jointLimits[static_cast<size_t>(i)][0];
    const double hi = jointLimits[static_cast<size_t>(i)][1];
    pTarget[i] = lo + uniDist(gen) * (hi - lo);
  }
  cfbFK(pTarget);
}

}  // namespace sweep_test_detail
}  // namespace raisim
