#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::Matrix;

using std::vector;


constexpr auto N_X = 5;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    weights_.setConstant(1 / (lambda_ + N_AUG) / 2.0);
    weights_(0) = lambda_ / (lambda_ + N_AUG);
}

UKF::~UKF()
{}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */
}


VectorD<N_X> UKF::ProcessModel(VectorD<N_AUG> x, double dt)
{
    VectorD<N_X> x_updated(5);
    double dt2_2 = dt * dt / 2.0;
    double v = x(2);
    double phi = x(3);
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    double phi_dot = x(4);
    double nu_a = x(5);
    double nu_phi_ddot = x(6);

    x_updated <<
              dt2_2 * cos_phi * nu_a,
        dt2_2 * sin_phi * nu_a,
        dt * nu_a,
        dt2_2 * nu_phi_ddot,
        dt * nu_phi_ddot;

    double delta_phi = phi_dot * dt;
    x_updated(3) += delta_phi;
    if (fabs(phi_dot) >= 0.000001)
    {
        double v_on_phi_dot = v / phi_dot;
        x_updated(0) += v_on_phi_dot * (sin(phi + delta_phi) - sin(phi));
        x_updated(1) += v_on_phi_dot * (-cos(phi + delta_phi) + cos(phi));
    }
    else
    {
        std::cout << phi_dot << std::endl;
        x_updated(0) += v * cos_phi * dt;
        x_updated(1) += v * sin_phi * dt;
    }
    x_updated += x.head(5);
    return x_updated;
}

VectorD<UKF::N_AUG> UKF::AugmentStateVector(VectorD<N_X> x)
{
    //create augmented mean state
    VectorD<N_AUG> x_aug;
    x_aug.head<N_X>() = x_;
    x_aug(N_X) = 0;
    x_aug(N_X + 1) = 0;

    return x_aug;
}

MatrixD<UKF::N_AUG> UKF::AugmentStateCovariance(MatrixD<N_X> P)
{
    //create augmented covariance matrix
    MatrixD<N_AUG> P_aug;
    P_aug.fill(0);
    P_aug.topLeftCorner<N_X, N_X>() = P_;
    P_aug(N_X, N_X) = std_a_ * std_a_;
    P_aug(N_X + 1, N_X + 1) = std_yawdd_ * std_yawdd_;

    return P_aug;
}

MatrixD<UKF::N_AUG, 2 * UKF::N_X + 1> UKF::AugmentedSamplePoints(VectorD<N_AUG> x_aug, MatrixD<N_AUG> P_aug)
{
    //create square root matrix
    MatrixD<N_AUG> A = P_aug.llt().matrixL();
    //create augmented sigma points
    MatrixD<N_AUG, 2 * N_X + 1> Xsig_aug;
    Xsig_aug.col(0) = x_aug;
    double scale = sqrt(lambda_ + N_AUG);
    for (size_t i = 0; i < N_AUG; ++i)
    {
        VectorD<N_AUG> col = A.col(i);
        Xsig_aug.col(i + 1) = x_aug + scale * col;
        Xsig_aug.col(i + 1 + N_AUG) = x_aug - scale * col;
    }
    return Xsig_aug;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    auto x_aug = AugmentStateVector(x_);
    auto P_aug = AugmentStateCovariance(P_);
    auto Xsig_aug = AugmentedSamplePoints(x_aug, P_aug);

    for(size_t i = 0; i < Xsig_pred_.cols(); ++i)
    {
        Xsig_pred_.col(i) = ProcessModel(Xsig_aug.col(i), delta_t);
    }

    //predict state mean
    x_ = Xsig_pred_ * weights_;
    //predict state covariance matrix
    Matrix<double, N_X, 2 * N_X + 1> err = Xsig_pred_.colwise() - x_;
    Matrix<double, N_X, 2 * N_X + 1> weightedErr = err.array().rowwise() * weights_.transpose().array();

    P_ = weightedErr * err.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
