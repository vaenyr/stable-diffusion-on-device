//==============================================================================
// Auto Generated Code for GroupNormPackage
//==============================================================================
#include <iostream>
#include <string>
#include <cmath>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace groupnorm {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto input = operation->getInput(0);
  auto weight_ptr = (float*)operation->getInput(1)->data;
  auto bias_ptr = (float*)operation->getInput(2)->data;
  auto output = operation->getOutput(0);
  auto input_ptr = (float*)input->data;
  auto output_ptr = (float*)output->data;

  uint32_t num_groups = (uint32_t)(operation->getParam("num_groups")->scalarParam);
  float eps = (float)(operation->getParam("eps")->scalarParam);

  uint32_t batch = (uint32_t)(input->currentDimensions[0]);
  uint32_t input_channels = (uint32_t)(input->currentDimensions[1]);
  uint32_t spatial = (uint32_t)(input->currentDimensions[2] * input->currentDimensions[3]);
  uint32_t channels_per_group = input_channels / num_groups;

  uint32_t total_groups = batch*num_groups;
  uint32_t group_size = channels_per_group*spatial;

  for (uint32_t b=0; b<batch; ++b) {
    for (uint32_t g=0; g<num_groups; ++g) {
      const float* this_input = input_ptr + (b*num_groups*group_size + g*group_size);
      float* this_output = output_ptr + (b*num_groups*group_size + g*group_size);
      uint32_t first_channel = g*channels_per_group;

      float mean = 0.0f;
      for (uint32_t el=0; el<group_size; ++el) {
        float tmp = this_input[el];
        mean += tmp;
        this_output[el] = tmp;
      }
      mean /= group_size;

      float var = 0.0f;
      for (uint32_t el=0; el<group_size; ++el) {
        float tmp = (this_input[el] - mean);
        var += tmp*tmp;
      }
      var /= group_size;

      var = sqrtf(var) + eps;

      for (uint32_t c=0; c<channels_per_group; ++c) {
        float w = weight_ptr[c+first_channel];
        float b = bias_ptr[c+first_channel];
        b += mean*w/var;
        w /= var;
        for (uint32_t s=0; s<spatial; ++s) {
          this_output[c*spatial + s] = this_output[c*spatial + s]*w - b;
        }
      }
    }
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  // Add params
   // The getParam function returns a pair -> hasParam, paramValue
   // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr

   auto num_groupsPair = getParam(node, "num_groups");

   QNN_CUSTOM_BE_ENSURE(num_groupsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("num_groups", num_groupsPair.second);


   auto epsPair = getParam(node, "eps");

   QNN_CUSTOM_BE_ENSURE(epsPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("eps", epsPair.second);


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "GroupNorm"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace groupnorm

CustomOpRegistration_t* register_GroupnormCustomOp() {
  using namespace groupnorm;
  static CustomOpRegistration_t GroupnormRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &GroupnormRegister;
}

REGISTER_OP(GroupNorm, register_GroupnormCustomOp);
