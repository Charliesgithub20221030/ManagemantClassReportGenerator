
Şý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ę­
|
gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
đ* 
shared_namegru_cell/kernel
u
#gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru_cell/kernel* 
_output_shapes
:
đ*
dtype0

gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namegru_cell/recurrent_kernel

-gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0
w
gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namegru_cell/bias
p
!gru_cell/bias/Read/ReadVariableOpReadVariableOpgru_cell/bias*
_output_shapes
:	*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
đ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
đ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:đ*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:đ*
dtype0

NoOpNoOp
Ą
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ü
valueŇBĎ BČ
w
cell
	dense
trainable_variables
regularization_losses
	variables
	keras_api

signatures
~

kernel
	recurrent_kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
#
0
	1

2
3
4
 
#
0
	1

2
3
4
­
trainable_variables
layer_regularization_losses
regularization_losses
metrics

layers
non_trainable_variables
layer_metrics
	variables
 
KI
VARIABLE_VALUEgru_cell/kernel&cell/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_cell/recurrent_kernel0cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEgru_cell/bias$cell/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1

2

0
	1

2
 
­
trainable_variables
	variables
layer_regularization_losses
regularization_losses

layers
non_trainable_variables
layer_metrics
metrics
IG
VARIABLE_VALUEdense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
layer_regularization_losses
regularization_losses

 layers
!non_trainable_variables
"layer_metrics
#metrics
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙(*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙(
ć
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gru_cell/biasgru_cell/kernelgru_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_78664
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ś
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#gru_cell/kernel/Read/ReadVariableOp-gru_cell/recurrent_kernel/Read/ReadVariableOp!gru_cell/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_79229
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegru_cell/kernelgru_cell/recurrent_kernelgru_cell/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_79256Ż
î
ţ
>__inference_rnn_layer_call_and_return_conditional_losses_78619

inputs	
gru_cell_78604
gru_cell_78606
gru_cell_78608
dense_78612
dense_78614
identity˘dense/StatefulPartitionedCall˘ gru_cell/StatefulPartitionedCalli
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŞ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceŚ
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0zeros:output:0gru_cell_78604gru_cell_78606gru_cell_78608*
Tin	
2*
Tout
2**
_output_shapes
:	2:	2*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_784922"
 gru_cell/StatefulPartitionedCallű
dense/StatefulPartitionedCallStatefulPartitionedCall)gru_cell/StatefulPartitionedCall:output:0dense_78612dense_78614*
Tin
2*
Tout
2*
_output_shapes
:	2đ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_785402
dense/StatefulPartitionedCallo
SoftmaxSoftmax&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:	2đ2	
Softmax 
IdentityIdentitySoftmax:softmax:0^dense/StatefulPartitionedCall!^gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ş
ĺ
C__inference_gru_cell_layer_call_and_return_conditional_losses_78492

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split/split_dimŻ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	22
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split_1/split_dimť
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	22
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	22	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	22
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	22
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	22
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	22
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	22
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	22
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	22
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	22
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	22

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ß
¨
@__inference_dense_layer_call_and_return_conditional_losses_78540

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2:::G C

_output_shapes
:	2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ě3
ľ
>__inference_rnn_layer_call_and_return_conditional_losses_78776
input_1	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŤ
one_hotOneHotinput_1one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ě3
ľ
>__inference_rnn_layer_call_and_return_conditional_losses_78720
input_1	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŤ
one_hotOneHotinput_1one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
É3
´
>__inference_rnn_layer_call_and_return_conditional_losses_78862

inputs	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŞ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ß
¨
@__inference_dense_layer_call_and_return_conditional_losses_79178

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2:::G C

_output_shapes
:	2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ú	
¨
(__inference_gru_cell_layer_call_fn_79168

inputs

states
unknown
	unknown_0
	unknown_1
identity

identity_1˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2**
_output_shapes
:	2:	2*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_784922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Š
 __inference__wrapped_model_78397
input_1	
	rnn_78385
	rnn_78387
	rnn_78389
	rnn_78391
	rnn_78393
identity˘rnn/StatefulPartitionedCallÎ
rnn/StatefulPartitionedCallStatefulPartitionedCallinput_1	rnn_78385	rnn_78387	rnn_78389	rnn_78391	rnn_78393*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*
fR
__inference_call_783842
rnn/StatefulPartitionedCall
IdentityIdentity$rnn/StatefulPartitionedCall:output:0^rnn/StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ú	
¨
(__inference_gru_cell_layer_call_fn_79154

inputs

states
unknown
	unknown_0
	unknown_1
identity

identity_1˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsstatesunknown	unknown_0	unknown_1*
Tin	
2*
Tout
2**
_output_shapes
:	2:	2*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_784522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	22

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Î
z
%__inference_dense_layer_call_fn_79187

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
:	2đ*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_785402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*&
_input_shapes
:	2::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ą3

__inference_call_78384

inputs	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŞ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú
Ś
#__inference_signature_wrapper_78664
input_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity˘StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_783972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ş
ĺ
C__inference_gru_cell_layer_call_and_return_conditional_losses_79140

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split/split_dimŻ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	22
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split_1/split_dimť
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	22
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	22	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	22
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	22
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	22
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	22
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	22
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	22
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	22
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	22
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	22

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ą3

__inference_call_79060

inputs	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŞ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ş
ĺ
C__inference_gru_cell_layer_call_and_return_conditional_losses_78452

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split/split_dimŻ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	22
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split_1/split_dimť
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	22
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	22	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	22
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	22
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	22
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	22
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	22
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	22
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	22
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	22
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	22

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ś
#__inference_rnn_layer_call_fn_78806
input_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity˘StatefulPartitionedCallě
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_786192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ş
ĺ
C__inference_gru_cell_layer_call_and_return_conditional_losses_79100

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split/split_dimŻ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	22
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
split_1/split_dimť
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	22
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	22	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	22
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	22
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	22
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	22
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	22
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	22
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	22
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	22
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	22

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙đ:	2::::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ
 
_user_specified_nameinputs:GC

_output_shapes
:	2
 
_user_specified_namestates:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ľ
#__inference_rnn_layer_call_fn_78948

inputs	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_786192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˛!
ń
__inference__traced_save_79229
file_prefix.
*savev2_gru_cell_kernel_read_readvariableop8
4savev2_gru_cell_recurrent_kernel_read_readvariableop,
(savev2_gru_cell_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_1_const

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_13e95b76ae0c405cb5148db3126b7c84/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ĺ
valueŰBŘB&cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB0cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB$cell/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_gru_cell_kernel_read_readvariableop4savev2_gru_cell_recurrent_kernel_read_readvariableop(savev2_gru_cell_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*M
_input_shapes<
:: :
đ:
:	:
đ:đ: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
đ:&"
 
_output_shapes
:
:%!

_output_shapes
:	:&"
 
_output_shapes
:
đ:!

_output_shapes	
:đ:

_output_shapes
: 
ż

!__inference__traced_restore_79256
file_prefix$
 assignvariableop_gru_cell_kernel0
,assignvariableop_1_gru_cell_recurrent_kernel$
 assignvariableop_2_gru_cell_bias#
assignvariableop_3_dense_kernel!
assignvariableop_4_dense_bias

identity_6˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘	RestoreV2˘RestoreV2_1Ů
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ĺ
valueŰBŘB&cell/kernel/.ATTRIBUTES/VARIABLE_VALUEB0cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB$cell/bias/.ATTRIBUTES/VARIABLE_VALUEB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesÄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_gru_cell_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1˘
AssignVariableOp_1AssignVariableOp,assignvariableop_1_gru_cell_recurrent_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp assignvariableop_2_gru_cell_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpĎ

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5Ű

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Đ2

__inference_call_79004

inputs	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthĄ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*#
_output_shapes
:2(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	2đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	22
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dim¸
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*1
_input_shapes 
:2(::::::F B

_output_shapes

:2(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ś
#__inference_rnn_layer_call_fn_78791
input_1	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity˘StatefulPartitionedCallě
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_786192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
É3
´
>__inference_rnn_layer_call_and_return_conditional_losses_78918

inputs	$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityi
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
one_hot/on_valuek
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
one_hot/off_valuea
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :đ2
one_hot/depthŞ
one_hotOneHotinputsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙(đ2	
one_hot
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"2      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	22
zeros
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceone_hot:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙đ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_cell/unstackŞ
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split/split_dimÓ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split2
gru_cell/split°
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	22
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ˙˙˙˙2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2
gru_cell/split_1/split_dimč
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	2:	2:	2*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	22
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	22
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	22
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	22
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	22
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	22
gru_cell/Tanhw
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*
_output_shapes
:	22
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	22
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	22
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	22
gru_cell/add_3Ą
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
đ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulgru_cell/add_3:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:đ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2đ2
dense/BiasAdd_
SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*
_output_shapes
:	2đ2	
Softmax]
IdentityIdentitySoftmax:softmax:0*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ľ
#__inference_rnn_layer_call_fn_78933

inputs	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity˘StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_output_shapes
:	2đ*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_786192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2đ2

Identity"
identityIdentity:output:0*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙(:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙(
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "ŻL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ł
serving_default
;
input_10
serving_default_input_1:0	˙˙˙˙˙˙˙˙˙(4
output_1(
StatefulPartitionedCall:0	2đtensorflow/serving/predict:ŚM
Ś
cell
	dense
trainable_variables
regularization_losses
	variables
	keras_api

signatures
$__call__
%_default_save_signature
*&&call_and_return_all_conditional_losses
'call"Ë
_tf_keras_modelą{"class_name": "RNN", "name": "rnn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "RNN"}}
Ă

kernel
	recurrent_kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layerî{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 1776]}}
Í

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*__call__
*+&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1776, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 256]}}
C
0
	1

2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
Ę
trainable_variables
layer_regularization_losses
regularization_losses
metrics

layers
non_trainable_variables
layer_metrics
	variables
$__call__
%_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
,
,serving_default"
signature_map
#:!
đ2gru_cell/kernel
-:+
2gru_cell/recurrent_kernel
 :	2gru_cell/bias
5
0
	1

2"
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
	variables
layer_regularization_losses
regularization_losses

layers
non_trainable_variables
layer_metrics
metrics
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 :
đ2dense/kernel
:đ2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
	variables
layer_regularization_losses
regularization_losses

 layers
!non_trainable_variables
"layer_metrics
#metrics
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ŕ2Ý
#__inference_rnn_layer_call_fn_78806
#__inference_rnn_layer_call_fn_78948
#__inference_rnn_layer_call_fn_78933
#__inference_rnn_layer_call_fn_78791Ć
˝˛š
FullArgSpec8
args0-
jself
jinputs
jfrom_logits

jtraining
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ţ2Ű
 __inference__wrapped_model_78397ś
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙(	
Ě2É
>__inference_rnn_layer_call_and_return_conditional_losses_78918
>__inference_rnn_layer_call_and_return_conditional_losses_78720
>__inference_rnn_layer_call_and_return_conditional_losses_78776
>__inference_rnn_layer_call_and_return_conditional_losses_78862Ć
˝˛š
FullArgSpec8
args0-
jself
jinputs
jfrom_logits

jtraining
varargs
 
varkw
 
defaults
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
__inference_call_79004
__inference_call_79060ś
­˛Š
FullArgSpec,
args$!
jself
jinputs
jfrom_logits
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
(__inference_gru_cell_layer_call_fn_79168
(__inference_gru_cell_layer_call_fn_79154ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Î2Ë
C__inference_gru_cell_layer_call_and_return_conditional_losses_79100
C__inference_gru_cell_layer_call_and_return_conditional_losses_79140ž
ľ˛ą
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ď2Ě
%__inference_dense_layer_call_fn_79187˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ę2ç
@__inference_dense_layer_call_and_return_conditional_losses_79178˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2B0
#__inference_signature_wrapper_78664input_1
 __inference__wrapped_model_78397f
	0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙(	
Ş "+Ş(
&
output_1
output_1	2đ_
__inference_call_79004E
	*˘'
 ˘

inputs2(	
p 
Ş "	2đh
__inference_call_79060N
	3˘0
)˘&
 
inputs˙˙˙˙˙˙˙˙˙(	
p 
Ş "	2đ
@__inference_dense_layer_call_and_return_conditional_losses_79178L'˘$
˘

inputs	2
Ş "˘

0	2đ
 h
%__inference_dense_layer_call_fn_79187?'˘$
˘

inputs	2
Ş "	2đá
C__inference_gru_cell_layer_call_and_return_conditional_losses_79100
	N˘K
D˘A
!
inputs˙˙˙˙˙˙˙˙˙đ

states	2
p
Ş "B˘?
8˘5

0/0	2


0/1/0	2
 á
C__inference_gru_cell_layer_call_and_return_conditional_losses_79140
	N˘K
D˘A
!
inputs˙˙˙˙˙˙˙˙˙đ

states	2
p 
Ş "B˘?
8˘5

0/0	2


0/1/0	2
 ¸
(__inference_gru_cell_layer_call_fn_79154
	N˘K
D˘A
!
inputs˙˙˙˙˙˙˙˙˙đ

states	2
p
Ş "4˘1

0	2


1/0	2¸
(__inference_gru_cell_layer_call_fn_79168
	N˘K
D˘A
!
inputs˙˙˙˙˙˙˙˙˙đ

states	2
p 
Ş "4˘1

0	2


1/0	2˘
>__inference_rnn_layer_call_and_return_conditional_losses_78720`
	8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙(	
p 
p
Ş "˘

0	2đ
 ˘
>__inference_rnn_layer_call_and_return_conditional_losses_78776`
	8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙(	
p 
p 
Ş "˘

0	2đ
 Ą
>__inference_rnn_layer_call_and_return_conditional_losses_78862_
	7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(	
p 
p
Ş "˘

0	2đ
 Ą
>__inference_rnn_layer_call_and_return_conditional_losses_78918_
	7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(	
p 
p 
Ş "˘

0	2đ
 z
#__inference_rnn_layer_call_fn_78791S
	8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙(	
p 
p
Ş "	2đz
#__inference_rnn_layer_call_fn_78806S
	8˘5
.˘+
!
input_1˙˙˙˙˙˙˙˙˙(	
p 
p 
Ş "	2đy
#__inference_rnn_layer_call_fn_78933R
	7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(	
p 
p
Ş "	2đy
#__inference_rnn_layer_call_fn_78948R
	7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙(	
p 
p 
Ş "	2đ
#__inference_signature_wrapper_78664q
	;˘8
˘ 
1Ş.
,
input_1!
input_1˙˙˙˙˙˙˙˙˙(	"+Ş(
&
output_1
output_1	2đ