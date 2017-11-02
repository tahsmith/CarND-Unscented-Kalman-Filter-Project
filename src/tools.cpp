#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd>& estimations,
                              const vector<VectorXd>& ground_truth)
{
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    size_t count = 0;
    for (int i = 0; i < estimations.size(); ++i)
    {
        VectorXd err = estimations[i] - ground_truth[i];
        if ((estimations[i].size() != ground_truth[i].size())
            || estimations[i].size() == 0)
        {
            continue;
        }
        VectorXd err_sq = err.array() * err.array();
        rmse += err_sq;
        count += 1;
    }

    //calculate the mean
    rmse = rmse / count;

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}