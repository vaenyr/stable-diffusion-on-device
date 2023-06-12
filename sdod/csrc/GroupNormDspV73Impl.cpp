//==============================================================================
// Auto Generated Code for GroupNormPackage
//==============================================================================
#include "optimize.h"
#include "op_register_ext.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_GroupNorm)

// op execute function declarations
template<typename TensorType>
int groupnormImpl(TensorType& out_0,
                  const TensorType& in_0,
                  const TensorType& weight,
                  const TensorType& bias,
                  const Tensor& num_groups,
                  const Tensor& eps);

//op definitions
DEF_PACKAGE_OP((groupnormImpl<Tensor>), "GroupNorm")

/* execute functions for ops */

template<typename TensorType>
int groupnormImpl(TensorType& out_0,
                  const TensorType& in_0,
                  const TensorType& weight,
                  const TensorType& bias,
                  const Tensor& num_groups,
                  const Tensor& eps)

{
/*
* add code here
           * */
return GraphStatus::Success;
}



END_PKG_OP_DEFINITION(PKG_GroupNorm)
