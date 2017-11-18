#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;
template <int N>
using VectorD = Eigen::Matrix<double, N, 1>;

template <int N, int N2=N>
using MatrixD = Eigen::Matrix<double, N, N2>;

class UKF
{
public:
    static constexpr size_t N_X = 5;
    static constexpr size_t N_AUG = 7;

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorD<N_X> x_;

    ///* state covariance matrix
    MatrixD<N_X> P_;

    ///* predicted sigma points matrix
    MatrixD<N_X, 2 * N_AUG + 1> Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    ///* Weights of sigma points
    VectorD<2 * N_AUG + 1> weights_;

    ///* Sigma point spreading parameter
    double lambda_;


    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     *
     */
    VectorD<N_X> ProcessModel(VectorD<N_AUG> x, double t);

    /**
     *
     * @param x
     * @return
     */
    VectorD<N_AUG> AugmentStateVector(VectorD<N_X> x);

    MatrixD<N_AUG> AugmentStateCovariance(MatrixD<N_X> P);

    MatrixD<N_AUG, 2 * N_AUG + 1>
    AugmentedSamplePoints(VectorD<N_AUG> x, MatrixD<N_AUG> P_aug);

    void KalmanUpdate(MatrixXd S, MatrixXd Zsig_diff, VectorXd z_diff);

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
    void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
