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
    x_.fill(0);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_.fill(0);

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

    lambda_ = 3 - N_AUG;
    Xsig_pred_.fill(0);
    weights_.setConstant(1 / (lambda_ + N_AUG) / 2.0);
    weights_(0) = lambda_ / (lambda_ + N_AUG);

    time_us_ = 0;
}

UKF::~UKF()
{}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    if (time_us_ == 0)
    {
        time_us_ = meas_package.timestamp_;
        return;
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        UpdateRadar(meas_package);
    }
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
    for (size_t i = 0; i < N_X; ++i)
    {
        VectorD<N_AUG> col = A.col(i);
        Xsig_aug.col(i + 1) = x_aug + scale * col;
        Xsig_aug.col(i + 1 + N_X) = x_aug - scale * col;
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

void UKF::KalmanUpdate(MatrixXd S, MatrixXd Zsig_pred, VectorXd z_pred, VectorXd z)
{
    MatrixD<N_X, 2 * N_X + 1> stateError = Xsig_pred_.colwise() - x_;
    MatrixD<N_X, 2 * N_X + 1> weighedStateError = stateError.array().rowwise() * weights_.transpose().array();
    MatrixXd measurementError = Zsig_pred.colwise() - z_pred;
    MatrixXd Tc = weighedStateError * measurementError.transpose();

    MatrixXd K = Tc * S.inverse();
    x_ += K * (z - z_pred);
    P_ += - K * S * K.transpose();
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
    using SigmaRow = Eigen::Matrix<double, 1, 2 * N_X + 1>;
    SigmaRow px = Xsig_pred_.row(0);
    SigmaRow py = Xsig_pred_.row(1);
    SigmaRow v = Xsig_pred_.row(2);
    SigmaRow phi = Xsig_pred_.row(3);
    SigmaRow phidot = Xsig_pred_.row(4);

    SigmaRow rho = (px.cwiseProduct(py) + py.cwiseProduct(py)).array().sqrt();
    MatrixD<3, 2 * N_X + 1> Zsig_pred;
    if ((rho.cwiseAbs().array() <= 0.0001).any())
    {
        return;
    }
    Zsig_pred.row(0) = rho;
    Zsig_pred.row(1) = py.binaryExpr(px, [] (double y, double x) { return atan2(y, x); });
    Zsig_pred.row(2) = (px.array() * phi.array().cos() + py.array() * phi.array().sin())
                       * v.array()
                       / rho.array();

    VectorD<3> z_pred;
    z_pred = Zsig_pred * weights_;
    //predict state covariance matrix
    VectorD<3> R;
    R <<
      std_radr_ * std_radr_,
        std_radphi_ * std_radphi_,
        std_radrd_ * std_radrd_;
    MatrixD<3, 2 * N_X + 1> err = Zsig_pred.colwise() - z_pred;
    MatrixD<3, 2 * N_X + 1> weightedErr = err.array().rowwise() * weights_.transpose().array();

    weightedErr += R.asDiagonal();
    MatrixD<3, 3> S = weightedErr * err.transpose();

    KalmanUpdate(S, Zsig_pred, z_pred, meas_package.raw_measurements_);
}
