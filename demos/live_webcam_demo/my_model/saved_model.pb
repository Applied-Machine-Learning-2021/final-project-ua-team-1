??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
: *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
: *
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:@*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?b?* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
?b?*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:?*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	?*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_24/kernel/m
?
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_25/kernel/m
?
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_26/kernel/m
?
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/m
{
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_27/kernel/m
?
+Adam/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_27/bias/m
{
)Adam/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?b?*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
?b?*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_24/kernel/v
?
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_25/kernel/v
?
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_26/kernel/v
?
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/v
{
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_27/kernel/v
?
+Adam/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_27/bias/v
{
)Adam/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?b?*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
?b?*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?\
value?[B?[ B?[
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
R
1	variables
2trainable_variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
R
_	variables
`trainable_variables
aregularization_losses
b	keras_api
h

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
R
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?5m?6m??m?@m?Um?Vm?cm?dm?v?v?#v?$v?5v?6v??v?@v?Uv?Vv?cv?dv?
V
0
1
#2
$3
54
65
?6
@7
U8
V9
c10
d11
V
0
1
#2
$3
54
65
?6
@7
U8
V9
c10
d11
 
?
	variables

rlayers
snon_trainable_variables
tmetrics
trainable_variables
ulayer_regularization_losses
vlayer_metrics
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables

wlayers
xnon_trainable_variables
ymetrics
trainable_variables
zlayer_regularization_losses
{layer_metrics
regularization_losses
 
 
 
?
	variables

|layers
}non_trainable_variables
~metrics
 trainable_variables
layer_regularization_losses
?layer_metrics
!regularization_losses
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
%	variables
?layers
?non_trainable_variables
?metrics
&trainable_variables
 ?layer_regularization_losses
?layer_metrics
'regularization_losses
 
 
 
?
)	variables
?layers
?non_trainable_variables
?metrics
*trainable_variables
 ?layer_regularization_losses
?layer_metrics
+regularization_losses
 
 
 
?
-	variables
?layers
?non_trainable_variables
?metrics
.trainable_variables
 ?layer_regularization_losses
?layer_metrics
/regularization_losses
 
 
 
?
1	variables
?layers
?non_trainable_variables
?metrics
2trainable_variables
 ?layer_regularization_losses
?layer_metrics
3regularization_losses
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
?
7	variables
?layers
?non_trainable_variables
?metrics
8trainable_variables
 ?layer_regularization_losses
?layer_metrics
9regularization_losses
 
 
 
?
;	variables
?layers
?non_trainable_variables
?metrics
<trainable_variables
 ?layer_regularization_losses
?layer_metrics
=regularization_losses
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
A	variables
?layers
?non_trainable_variables
?metrics
Btrainable_variables
 ?layer_regularization_losses
?layer_metrics
Cregularization_losses
 
 
 
?
E	variables
?layers
?non_trainable_variables
?metrics
Ftrainable_variables
 ?layer_regularization_losses
?layer_metrics
Gregularization_losses
 
 
 
?
I	variables
?layers
?non_trainable_variables
?metrics
Jtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Kregularization_losses
 
 
 
?
M	variables
?layers
?non_trainable_variables
?metrics
Ntrainable_variables
 ?layer_regularization_losses
?layer_metrics
Oregularization_losses
 
 
 
?
Q	variables
?layers
?non_trainable_variables
?metrics
Rtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Sregularization_losses
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
?
W	variables
?layers
?non_trainable_variables
?metrics
Xtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Yregularization_losses
 
 
 
?
[	variables
?layers
?non_trainable_variables
?metrics
\trainable_variables
 ?layer_regularization_losses
?layer_metrics
]regularization_losses
 
 
 
?
_	variables
?layers
?non_trainable_variables
?metrics
`trainable_variables
 ?layer_regularization_losses
?layer_metrics
aregularization_losses
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
?
e	variables
?layers
?non_trainable_variables
?metrics
ftrainable_variables
 ?layer_regularization_losses
?layer_metrics
gregularization_losses
 
 
 
?
i	variables
?layers
?non_trainable_variables
?metrics
jtrainable_variables
 ?layer_regularization_losses
?layer_metrics
kregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
 

?0
?1
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
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_24/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_27/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_27/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_27/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_27/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_24_inputPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_24_inputconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_17591
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp+Adam/conv2d_27/kernel/m/Read/ReadVariableOp)Adam/conv2d_27/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp+Adam/conv2d_27/kernel/v/Read/ReadVariableOp)Adam/conv2d_27/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_18200
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/conv2d_27/kernel/mAdam/conv2d_27/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/conv2d_27/kernel/vAdam/conv2d_27/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_18345??

?
?
(__inference_dense_13_layer_call_fn_18022

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_171172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_13_layer_call_fn_16954

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_169482
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_39_layer_call_and_return_conditional_losses_17919

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_17957

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????b2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_flatten_6_layer_call_fn_17951

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_170752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_17824

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
I
-__inference_activation_36_layer_call_fn_17800

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_36_layer_call_and_return_conditional_losses_169822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
d
H__inference_activation_40_layer_call_and_return_conditional_losses_17986

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_19_layer_call_and_return_conditional_losses_17239

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_16936

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?G
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17508
conv2d_24_input)
conv2d_24_17465: 
conv2d_24_17467: )
conv2d_25_17471:  
conv2d_25_17473: )
conv2d_26_17479: @
conv2d_26_17481:@)
conv2d_27_17485:@@
conv2d_27_17487:@"
dense_12_17494:
?b?
dense_12_17496:	?!
dense_13_17501:	?
dense_13_17503:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputconv2d_24_17465conv2d_24_17467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_169712#
!conv2d_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_36_layer_call_and_return_conditional_losses_169822
activation_36/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_25_17471conv2d_25_17473*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_169942#
!conv2d_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_170052
activation_37/PartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_169362"
 max_pooling2d_12/PartitionedCall?
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_170132
dropout_18/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_26_17479conv2d_26_17481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_170252#
!conv2d_26/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_38_layer_call_and_return_conditional_losses_170362
activation_38/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0conv2d_27_17485conv2d_27_17487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_170482#
!conv2d_27/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_170592
activation_39/PartitionedCall?
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_169482"
 max_pooling2d_13/PartitionedCall?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_170672
dropout_19/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall#dropout_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_170752
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_17494dense_12_17496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_170872"
 dense_12/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_170982
activation_40/PartitionedCall?
dropout_20/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171052
dropout_20/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_13_17501dense_13_17503*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_171172"
 dense_13/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_171282
activation_41/PartitionedCall?
IdentityIdentity&activation_41/PartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?
d
H__inference_activation_38_layer_call_and_return_conditional_losses_17036

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_17067

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_27_layer_call_fn_17899

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_170482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_38_layer_call_fn_17885

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_38_layer_call_and_return_conditional_losses_170362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?L
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17554
conv2d_24_input)
conv2d_24_17511: 
conv2d_24_17513: )
conv2d_25_17517:  
conv2d_25_17519: )
conv2d_26_17525: @
conv2d_26_17527:@)
conv2d_27_17531:@@
conv2d_27_17533:@"
dense_12_17540:
?b?
dense_12_17542:	?!
dense_13_17547:	?
dense_13_17549:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputconv2d_24_17511conv2d_24_17513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_169712#
!conv2d_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_36_layer_call_and_return_conditional_losses_169822
activation_36/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_25_17517conv2d_25_17519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_169942#
!conv2d_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_170052
activation_37/PartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_169362"
 max_pooling2d_12/PartitionedCall?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_172942$
"dropout_18/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_26_17525conv2d_26_17527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_170252#
!conv2d_26/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_38_layer_call_and_return_conditional_losses_170362
activation_38/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0conv2d_27_17531conv2d_27_17533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_170482#
!conv2d_27/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_170592
activation_39/PartitionedCall?
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_169482"
 max_pooling2d_13/PartitionedCall?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_172392$
"dropout_19/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall+dropout_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_170752
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_17540dense_12_17542*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_170872"
 dense_12/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_170982
activation_40/PartitionedCall?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall&activation_40/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171942$
"dropout_20/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_13_17547dense_13_17549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_171172"
 dense_13/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_171282
activation_41/PartitionedCall?
IdentityIdentity&activation_41/PartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?
d
H__inference_activation_40_layer_call_and_return_conditional_losses_17098

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?	
G__inference_sequential_6_layer_call_and_return_conditional_losses_17702

inputsB
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: B
(conv2d_25_conv2d_readvariableop_resource:  7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: @7
)conv2d_26_biasadd_readvariableop_resource:@B
(conv2d_27_conv2d_readvariableop_resource:@@7
)conv2d_27_biasadd_readvariableop_resource:@;
'dense_12_matmul_readvariableop_resource:
?b?7
(dense_12_biasadd_readvariableop_resource:	?:
'dense_13_matmul_readvariableop_resource:	?6
(dense_13_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_24/BiasAdd?
activation_36/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
activation_36/Relu?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D activation_36/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> *
paddingVALID*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> 2
conv2d_25/BiasAdd?
activation_37/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>> 2
activation_37/Relu?
max_pooling2d_12/MaxPoolMaxPool activation_37/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool?
dropout_18/IdentityIdentity!max_pooling2d_12/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_18/Identity?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Ddropout_18/Identity:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/BiasAdd?
activation_38/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
activation_38/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D activation_38/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/BiasAdd?
activation_39/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
activation_39/Relu?
max_pooling2d_13/MaxPoolMaxPool activation_39/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPool?
dropout_19/IdentityIdentity!max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_19/Identitys
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_19/Identity:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_6/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
?b?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd~
activation_40/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_40/Relu?
dropout_20/IdentityIdentity activation_40/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_20/Identity?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_20/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?
activation_41/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_41/Softmax?
IdentityIdentityactivation_41/Softmax:softmax:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_17117

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?L
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17406

inputs)
conv2d_24_17363: 
conv2d_24_17365: )
conv2d_25_17369:  
conv2d_25_17371: )
conv2d_26_17377: @
conv2d_26_17379:@)
conv2d_27_17383:@@
conv2d_27_17385:@"
dense_12_17392:
?b?
dense_12_17394:	?!
dense_13_17399:	?
dense_13_17401:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_17363conv2d_24_17365*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_169712#
!conv2d_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_36_layer_call_and_return_conditional_losses_169822
activation_36/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_25_17369conv2d_25_17371*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_169942#
!conv2d_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_170052
activation_37/PartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_169362"
 max_pooling2d_12/PartitionedCall?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_172942$
"dropout_18/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_26_17377conv2d_26_17379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_170252#
!conv2d_26/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_38_layer_call_and_return_conditional_losses_170362
activation_38/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0conv2d_27_17383conv2d_27_17385*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_170482#
!conv2d_27/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_170592
activation_39/PartitionedCall?
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_169482"
 max_pooling2d_13/PartitionedCall?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_172392$
"dropout_19/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall+dropout_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_170752
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_17392dense_12_17394*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_170872"
 dense_12/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_170982
activation_40/PartitionedCall?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall&activation_40/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171942$
"dropout_20/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0dense_13_17399dense_13_17401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_171172"
 dense_13/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_171282
activation_41/PartitionedCall?
IdentityIdentity&activation_41/PartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
c
*__inference_dropout_18_layer_call_fn_17844

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_172942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_17795

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_17105

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17131

inputs)
conv2d_24_16972: 
conv2d_24_16974: )
conv2d_25_16995:  
conv2d_25_16997: )
conv2d_26_17026: @
conv2d_26_17028:@)
conv2d_27_17049:@@
conv2d_27_17051:@"
dense_12_17088:
?b?
dense_12_17090:	?!
dense_13_17118:	?
dense_13_17120:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24_16972conv2d_24_16974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_169712#
!conv2d_24/StatefulPartitionedCall?
activation_36/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_36_layer_call_and_return_conditional_losses_169822
activation_36/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall&activation_36/PartitionedCall:output:0conv2d_25_16995conv2d_25_16997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_169942#
!conv2d_25/StatefulPartitionedCall?
activation_37/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_170052
activation_37/PartitionedCall?
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_169362"
 max_pooling2d_12/PartitionedCall?
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_170132
dropout_18/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_26_17026conv2d_26_17028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_170252#
!conv2d_26/StatefulPartitionedCall?
activation_38/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_38_layer_call_and_return_conditional_losses_170362
activation_38/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall&activation_38/PartitionedCall:output:0conv2d_27_17049conv2d_27_17051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_170482#
!conv2d_27/StatefulPartitionedCall?
activation_39/PartitionedCallPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_170592
activation_39/PartitionedCall?
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_169482"
 max_pooling2d_13/PartitionedCall?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_170672
dropout_19/PartitionedCall?
flatten_6/PartitionedCallPartitionedCall#dropout_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_170752
flatten_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_17088dense_12_17090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_170872"
 dense_12/StatefulPartitionedCall?
activation_40/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_170982
activation_40/PartitionedCall?
dropout_20/PartitionedCallPartitionedCall&activation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171052
dropout_20/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0dense_13_17118dense_13_17120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_171172"
 dense_13/StatefulPartitionedCall?
activation_41/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_171282
activation_41/PartitionedCall?
IdentityIdentity&activation_41/PartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_17880

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_17025

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_17075

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????b2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_dropout_19_layer_call_fn_17924

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_170672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_sequential_6_layer_call_fn_17158
conv2d_24_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
?b?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_171312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?
d
H__inference_activation_37_layer_call_and_return_conditional_losses_17834

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????>> 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>> :W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?	
?
C__inference_dense_13_layer_call_and_return_conditional_losses_18032

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_36_layer_call_and_return_conditional_losses_17805

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
I
-__inference_activation_40_layer_call_fn_17981

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_40_layer_call_and_return_conditional_losses_170982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_12_layer_call_fn_17966

inputs
unknown:
?b?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_170872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_17013

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
,__inference_sequential_6_layer_call_fn_17649

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
?b?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_174062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
I
-__inference_activation_39_layer_call_fn_17914

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_170592
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_26_layer_call_fn_17870

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_170252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?^
?
__inference__traced_save_18200
file_prefix/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop6
2savev2_adam_conv2d_27_kernel_m_read_readvariableop4
0savev2_adam_conv2d_27_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop6
2savev2_adam_conv2d_27_kernel_v_read_readvariableop4
0savev2_adam_conv2d_27_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop2savev2_adam_conv2d_27_kernel_m_read_readvariableop0savev2_adam_conv2d_27_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop2savev2_adam_conv2d_27_kernel_v_read_readvariableop0savev2_adam_conv2d_27_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : : @:@:@@:@:
?b?:?:	?:: : : : : : : : : : : :  : : @:@:@@:@:
?b?:?:	?:: : :  : : @:@:@@:@:
?b?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&	"
 
_output_shapes
:
?b?:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
?b?:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@:,((
&
_output_shapes
:@@: )

_output_shapes
:@:&*"
 
_output_shapes
:
?b?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::.

_output_shapes
: 
?
?
,__inference_sequential_6_layer_call_fn_17462
conv2d_24_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
?b?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_174062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_17087

inputs2
matmul_readvariableop_resource:
?b?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?b?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
I
-__inference_activation_37_layer_call_fn_17829

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_170052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>> :W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?
c
*__inference_dropout_19_layer_call_fn_17929

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_172392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_17909

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
,__inference_sequential_6_layer_call_fn_17620

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
?b?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_171312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_12_layer_call_fn_16942

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_169362
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_20_layer_call_fn_17991

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_17934

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_dropout_18_layer_call_fn_17839

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_170132
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_20_layer_call_and_return_conditional_losses_17194

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_17048

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_17128

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?d
?	
G__inference_sequential_6_layer_call_and_return_conditional_losses_17776

inputsB
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: B
(conv2d_25_conv2d_readvariableop_resource:  7
)conv2d_25_biasadd_readvariableop_resource: B
(conv2d_26_conv2d_readvariableop_resource: @7
)conv2d_26_biasadd_readvariableop_resource:@B
(conv2d_27_conv2d_readvariableop_resource:@@7
)conv2d_27_biasadd_readvariableop_resource:@;
'dense_12_matmul_readvariableop_resource:
?b?7
(dense_12_biasadd_readvariableop_resource:	?:
'dense_13_matmul_readvariableop_resource:	?6
(dense_13_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2Dinputs'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2
conv2d_24/BiasAdd?
activation_36/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2
activation_36/Relu?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D activation_36/Relu:activations:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> *
paddingVALID*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> 2
conv2d_25/BiasAdd?
activation_37/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>> 2
activation_37/Relu?
max_pooling2d_12/MaxPoolMaxPool activation_37/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPooly
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_18/dropout/Const?
dropout_18/dropout/MulMul!max_pooling2d_12/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_18/dropout/Mul?
dropout_18/dropout/ShapeShape!max_pooling2d_12/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape?
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype021
/dropout_18/dropout/random_uniform/RandomUniform?
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_18/dropout/GreaterEqual/y?
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2!
dropout_18/dropout/GreaterEqual?
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_18/dropout/Cast?
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_18/dropout/Mul_1?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Ddropout_18/dropout/Mul_1:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/BiasAdd?
activation_38/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
activation_38/Relu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D activation_38/Relu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/BiasAdd?
activation_39/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
activation_39/Relu?
max_pooling2d_13/MaxPoolMaxPool activation_39/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPooly
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_19/dropout/Const?
dropout_19/dropout/MulMul!max_pooling2d_13/MaxPool:output:0!dropout_19/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_19/dropout/Mul?
dropout_19/dropout/ShapeShape!max_pooling2d_13/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape?
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform?
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_19/dropout/GreaterEqual/y?
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2!
dropout_19/dropout/GreaterEqual?
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_19/dropout/Cast?
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_19/dropout/Mul_1s
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_6/Const?
flatten_6/ReshapeReshapedropout_19/dropout/Mul_1:z:0flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_6/Reshape?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
?b?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd~
activation_40/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
activation_40/Reluy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_20/dropout/Const?
dropout_20/dropout/MulMul activation_40/Relu:activations:0!dropout_20/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_20/dropout/Mul?
dropout_20/dropout/ShapeShape activation_40/Relu:activations:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shape?
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform?
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_20/dropout/GreaterEqual/y?
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_20/dropout/GreaterEqual?
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_20/dropout/Cast?
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_20/dropout/Mul_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_20/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd?
activation_41/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_41/Softmax?
IdentityIdentityactivation_41/Softmax:softmax:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_18345
file_prefix;
!assignvariableop_conv2d_24_kernel: /
!assignvariableop_1_conv2d_24_bias: =
#assignvariableop_2_conv2d_25_kernel:  /
!assignvariableop_3_conv2d_25_bias: =
#assignvariableop_4_conv2d_26_kernel: @/
!assignvariableop_5_conv2d_26_bias:@=
#assignvariableop_6_conv2d_27_kernel:@@/
!assignvariableop_7_conv2d_27_bias:@6
"assignvariableop_8_dense_12_kernel:
?b?/
 assignvariableop_9_dense_12_bias:	?6
#assignvariableop_10_dense_13_kernel:	?/
!assignvariableop_11_dense_13_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: E
+assignvariableop_21_adam_conv2d_24_kernel_m: 7
)assignvariableop_22_adam_conv2d_24_bias_m: E
+assignvariableop_23_adam_conv2d_25_kernel_m:  7
)assignvariableop_24_adam_conv2d_25_bias_m: E
+assignvariableop_25_adam_conv2d_26_kernel_m: @7
)assignvariableop_26_adam_conv2d_26_bias_m:@E
+assignvariableop_27_adam_conv2d_27_kernel_m:@@7
)assignvariableop_28_adam_conv2d_27_bias_m:@>
*assignvariableop_29_adam_dense_12_kernel_m:
?b?7
(assignvariableop_30_adam_dense_12_bias_m:	?=
*assignvariableop_31_adam_dense_13_kernel_m:	?6
(assignvariableop_32_adam_dense_13_bias_m:E
+assignvariableop_33_adam_conv2d_24_kernel_v: 7
)assignvariableop_34_adam_conv2d_24_bias_v: E
+assignvariableop_35_adam_conv2d_25_kernel_v:  7
)assignvariableop_36_adam_conv2d_25_bias_v: E
+assignvariableop_37_adam_conv2d_26_kernel_v: @7
)assignvariableop_38_adam_conv2d_26_bias_v:@E
+assignvariableop_39_adam_conv2d_27_kernel_v:@@7
)assignvariableop_40_adam_conv2d_27_bias_v:@>
*assignvariableop_41_adam_dense_12_kernel_v:
?b?7
(assignvariableop_42_adam_dense_12_bias_v:	?=
*assignvariableop_43_adam_dense_13_kernel_v:	?6
(assignvariableop_44_adam_dense_13_bias_v:
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_25_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_26_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_26_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_24_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_24_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_25_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_25_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_26_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_26_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_27_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_27_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_12_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_12_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_13_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_13_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_24_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_24_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_25_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_25_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_26_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_26_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_27_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_27_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_12_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_12_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_13_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_13_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
d
E__inference_dropout_18_layer_call_and_return_conditional_losses_17861

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_12_layer_call_and_return_conditional_losses_17976

inputs2
matmul_readvariableop_resource:
?b?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?b?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_18042

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_18001

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_16948

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_24_layer_call_fn_17785

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_169712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
d
E__inference_dropout_18_layer_call_and_return_conditional_losses_17294

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?Y
?
 __inference__wrapped_model_16930
conv2d_24_inputO
5sequential_6_conv2d_24_conv2d_readvariableop_resource: D
6sequential_6_conv2d_24_biasadd_readvariableop_resource: O
5sequential_6_conv2d_25_conv2d_readvariableop_resource:  D
6sequential_6_conv2d_25_biasadd_readvariableop_resource: O
5sequential_6_conv2d_26_conv2d_readvariableop_resource: @D
6sequential_6_conv2d_26_biasadd_readvariableop_resource:@O
5sequential_6_conv2d_27_conv2d_readvariableop_resource:@@D
6sequential_6_conv2d_27_biasadd_readvariableop_resource:@H
4sequential_6_dense_12_matmul_readvariableop_resource:
?b?D
5sequential_6_dense_12_biasadd_readvariableop_resource:	?G
4sequential_6_dense_13_matmul_readvariableop_resource:	?C
5sequential_6_dense_13_biasadd_readvariableop_resource:
identity??-sequential_6/conv2d_24/BiasAdd/ReadVariableOp?,sequential_6/conv2d_24/Conv2D/ReadVariableOp?-sequential_6/conv2d_25/BiasAdd/ReadVariableOp?,sequential_6/conv2d_25/Conv2D/ReadVariableOp?-sequential_6/conv2d_26/BiasAdd/ReadVariableOp?,sequential_6/conv2d_26/Conv2D/ReadVariableOp?-sequential_6/conv2d_27/BiasAdd/ReadVariableOp?,sequential_6/conv2d_27/Conv2D/ReadVariableOp?,sequential_6/dense_12/BiasAdd/ReadVariableOp?+sequential_6/dense_12/MatMul/ReadVariableOp?,sequential_6/dense_13/BiasAdd/ReadVariableOp?+sequential_6/dense_13/MatMul/ReadVariableOp?
,sequential_6/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_6/conv2d_24/Conv2D/ReadVariableOp?
sequential_6/conv2d_24/Conv2DConv2Dconv2d_24_input4sequential_6/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
sequential_6/conv2d_24/Conv2D?
-sequential_6/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp?
sequential_6/conv2d_24/BiasAddBiasAdd&sequential_6/conv2d_24/Conv2D:output:05sequential_6/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2 
sequential_6/conv2d_24/BiasAdd?
sequential_6/activation_36/ReluRelu'sequential_6/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@ 2!
sequential_6/activation_36/Relu?
,sequential_6/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_6/conv2d_25/Conv2D/ReadVariableOp?
sequential_6/conv2d_25/Conv2DConv2D-sequential_6/activation_36/Relu:activations:04sequential_6/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> *
paddingVALID*
strides
2
sequential_6/conv2d_25/Conv2D?
-sequential_6/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp?
sequential_6/conv2d_25/BiasAddBiasAdd&sequential_6/conv2d_25/Conv2D:output:05sequential_6/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> 2 
sequential_6/conv2d_25/BiasAdd?
sequential_6/activation_37/ReluRelu'sequential_6/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????>> 2!
sequential_6/activation_37/Relu?
%sequential_6/max_pooling2d_12/MaxPoolMaxPool-sequential_6/activation_37/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_12/MaxPool?
 sequential_6/dropout_18/IdentityIdentity.sequential_6/max_pooling2d_12/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2"
 sequential_6/dropout_18/Identity?
,sequential_6/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_6/conv2d_26/Conv2D/ReadVariableOp?
sequential_6/conv2d_26/Conv2DConv2D)sequential_6/dropout_18/Identity:output:04sequential_6/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_6/conv2d_26/Conv2D?
-sequential_6/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp?
sequential_6/conv2d_26/BiasAddBiasAdd&sequential_6/conv2d_26/Conv2D:output:05sequential_6/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_6/conv2d_26/BiasAdd?
sequential_6/activation_38/ReluRelu'sequential_6/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2!
sequential_6/activation_38/Relu?
,sequential_6/conv2d_27/Conv2D/ReadVariableOpReadVariableOp5sequential_6_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_6/conv2d_27/Conv2D/ReadVariableOp?
sequential_6/conv2d_27/Conv2DConv2D-sequential_6/activation_38/Relu:activations:04sequential_6/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_6/conv2d_27/Conv2D?
-sequential_6/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_6_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp?
sequential_6/conv2d_27/BiasAddBiasAdd&sequential_6/conv2d_27/Conv2D:output:05sequential_6/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_6/conv2d_27/BiasAdd?
sequential_6/activation_39/ReluRelu'sequential_6/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2!
sequential_6/activation_39/Relu?
%sequential_6/max_pooling2d_13/MaxPoolMaxPool-sequential_6/activation_39/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2'
%sequential_6/max_pooling2d_13/MaxPool?
 sequential_6/dropout_19/IdentityIdentity.sequential_6/max_pooling2d_13/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2"
 sequential_6/dropout_19/Identity?
sequential_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
sequential_6/flatten_6/Const?
sequential_6/flatten_6/ReshapeReshape)sequential_6/dropout_19/Identity:output:0%sequential_6/flatten_6/Const:output:0*
T0*(
_output_shapes
:??????????b2 
sequential_6/flatten_6/Reshape?
+sequential_6/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
?b?*
dtype02-
+sequential_6/dense_12/MatMul/ReadVariableOp?
sequential_6/dense_12/MatMulMatMul'sequential_6/flatten_6/Reshape:output:03sequential_6/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_12/MatMul?
,sequential_6/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_6/dense_12/BiasAdd/ReadVariableOp?
sequential_6/dense_12/BiasAddBiasAdd&sequential_6/dense_12/MatMul:product:04sequential_6/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_12/BiasAdd?
sequential_6/activation_40/ReluRelu&sequential_6/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2!
sequential_6/activation_40/Relu?
 sequential_6/dropout_20/IdentityIdentity-sequential_6/activation_40/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 sequential_6/dropout_20/Identity?
+sequential_6/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_6_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_6/dense_13/MatMul/ReadVariableOp?
sequential_6/dense_13/MatMulMatMul)sequential_6/dropout_20/Identity:output:03sequential_6/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_13/MatMul?
,sequential_6/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_6_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_6/dense_13/BiasAdd/ReadVariableOp?
sequential_6/dense_13/BiasAddBiasAdd&sequential_6/dense_13/MatMul:product:04sequential_6/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_6/dense_13/BiasAdd?
"sequential_6/activation_41/SoftmaxSoftmax&sequential_6/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2$
"sequential_6/activation_41/Softmax?
IdentityIdentity,sequential_6/activation_41/Softmax:softmax:0.^sequential_6/conv2d_24/BiasAdd/ReadVariableOp-^sequential_6/conv2d_24/Conv2D/ReadVariableOp.^sequential_6/conv2d_25/BiasAdd/ReadVariableOp-^sequential_6/conv2d_25/Conv2D/ReadVariableOp.^sequential_6/conv2d_26/BiasAdd/ReadVariableOp-^sequential_6/conv2d_26/Conv2D/ReadVariableOp.^sequential_6/conv2d_27/BiasAdd/ReadVariableOp-^sequential_6/conv2d_27/Conv2D/ReadVariableOp-^sequential_6/dense_12/BiasAdd/ReadVariableOp,^sequential_6/dense_12/MatMul/ReadVariableOp-^sequential_6/dense_13/BiasAdd/ReadVariableOp,^sequential_6/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 2^
-sequential_6/conv2d_24/BiasAdd/ReadVariableOp-sequential_6/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_24/Conv2D/ReadVariableOp,sequential_6/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_25/BiasAdd/ReadVariableOp-sequential_6/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_25/Conv2D/ReadVariableOp,sequential_6/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_26/BiasAdd/ReadVariableOp-sequential_6/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_26/Conv2D/ReadVariableOp,sequential_6/conv2d_26/Conv2D/ReadVariableOp2^
-sequential_6/conv2d_27/BiasAdd/ReadVariableOp-sequential_6/conv2d_27/BiasAdd/ReadVariableOp2\
,sequential_6/conv2d_27/Conv2D/ReadVariableOp,sequential_6/conv2d_27/Conv2D/ReadVariableOp2\
,sequential_6/dense_12/BiasAdd/ReadVariableOp,sequential_6/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_12/MatMul/ReadVariableOp+sequential_6/dense_12/MatMul/ReadVariableOp2\
,sequential_6/dense_13/BiasAdd/ReadVariableOp,sequential_6/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_6/dense_13/MatMul/ReadVariableOp+sequential_6/dense_13/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_17849

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_20_layer_call_and_return_conditional_losses_18013

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_20_layer_call_fn_17996

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_171942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_37_layer_call_and_return_conditional_losses_17005

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????>> 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>> :W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?
d
H__inference_activation_36_layer_call_and_return_conditional_losses_16982

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_16971

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@ 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
d
E__inference_dropout_19_layer_call_and_return_conditional_losses_17946

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_38_layer_call_and_return_conditional_losses_17890

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_39_layer_call_and_return_conditional_losses_17059

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_25_layer_call_fn_17814

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_169942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_17591
conv2d_24_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
?b?
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_169302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????@@
)
_user_specified_nameconv2d_24_input
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_16994

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>> 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????>> 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
I
-__inference_activation_41_layer_call_fn_18037

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_171282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_24_input@
!serving_default_conv2d_24_input:0?????????@@A
activation_410
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?n
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?j
_tf_keras_sequential?j{"name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_24_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 21, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "float32", "conv2d_24_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_24_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Activation", "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7}, {"class_name": "Activation", "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 8}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "Activation", "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}, {"class_name": "Activation", "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 18}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 20}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 26}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 21, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "softmax"}, "shared_object_id": 30}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 33}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_36", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 4}
?


#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_37", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 8}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 35}}
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 10}
?


5kernel
6bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 32]}}
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_38", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 14}
?


?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 64]}}
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_39", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 18}
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 38}}
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 20}
?
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 39}}
?

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12544}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12544]}}
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_40", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 25}
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 26}
?

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 21, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "activation_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "activation_41", "trainable": true, "dtype": "float32", "activation": "softmax"}, "shared_object_id": 30}
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?5m?6m??m?@m?Um?Vm?cm?dm?v?v?#v?$v?5v?6v??v?@v?Uv?Vv?cv?dv?"
	optimizer
v
0
1
#2
$3
54
65
?6
@7
U8
V9
c10
d11"
trackable_list_wrapper
v
0
1
#2
$3
54
65
?6
@7
U8
V9
c10
d11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

rlayers
snon_trainable_variables
tmetrics
trainable_variables
ulayer_regularization_losses
vlayer_metrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_24/kernel
: 2conv2d_24/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

wlayers
xnon_trainable_variables
ymetrics
trainable_variables
zlayer_regularization_losses
{layer_metrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

|layers
}non_trainable_variables
~metrics
 trainable_variables
layer_regularization_losses
?layer_metrics
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_25/kernel
: 2conv2d_25/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%	variables
?layers
?non_trainable_variables
?metrics
&trainable_variables
 ?layer_regularization_losses
?layer_metrics
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)	variables
?layers
?non_trainable_variables
?metrics
*trainable_variables
 ?layer_regularization_losses
?layer_metrics
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-	variables
?layers
?non_trainable_variables
?metrics
.trainable_variables
 ?layer_regularization_losses
?layer_metrics
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1	variables
?layers
?non_trainable_variables
?metrics
2trainable_variables
 ?layer_regularization_losses
?layer_metrics
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_26/kernel
:@2conv2d_26/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7	variables
?layers
?non_trainable_variables
?metrics
8trainable_variables
 ?layer_regularization_losses
?layer_metrics
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
?layers
?non_trainable_variables
?metrics
<trainable_variables
 ?layer_regularization_losses
?layer_metrics
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_27/kernel
:@2conv2d_27/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
A	variables
?layers
?non_trainable_variables
?metrics
Btrainable_variables
 ?layer_regularization_losses
?layer_metrics
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
E	variables
?layers
?non_trainable_variables
?metrics
Ftrainable_variables
 ?layer_regularization_losses
?layer_metrics
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
I	variables
?layers
?non_trainable_variables
?metrics
Jtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
M	variables
?layers
?non_trainable_variables
?metrics
Ntrainable_variables
 ?layer_regularization_losses
?layer_metrics
Oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Q	variables
?layers
?non_trainable_variables
?metrics
Rtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?b?2dense_12/kernel
:?2dense_12/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
W	variables
?layers
?non_trainable_variables
?metrics
Xtrainable_variables
 ?layer_regularization_losses
?layer_metrics
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[	variables
?layers
?non_trainable_variables
?metrics
\trainable_variables
 ?layer_regularization_losses
?layer_metrics
]regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_	variables
?layers
?non_trainable_variables
?metrics
`trainable_variables
 ?layer_regularization_losses
?layer_metrics
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_13/kernel
:2dense_13/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
e	variables
?layers
?non_trainable_variables
?metrics
ftrainable_variables
 ?layer_regularization_losses
?layer_metrics
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
i	variables
?layers
?non_trainable_variables
?metrics
jtrainable_variables
 ?layer_regularization_losses
?layer_metrics
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 42}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 33}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_24/kernel/m
!: 2Adam/conv2d_24/bias/m
/:-  2Adam/conv2d_25/kernel/m
!: 2Adam/conv2d_25/bias/m
/:- @2Adam/conv2d_26/kernel/m
!:@2Adam/conv2d_26/bias/m
/:-@@2Adam/conv2d_27/kernel/m
!:@2Adam/conv2d_27/bias/m
(:&
?b?2Adam/dense_12/kernel/m
!:?2Adam/dense_12/bias/m
':%	?2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
/:- 2Adam/conv2d_24/kernel/v
!: 2Adam/conv2d_24/bias/v
/:-  2Adam/conv2d_25/kernel/v
!: 2Adam/conv2d_25/bias/v
/:- @2Adam/conv2d_26/kernel/v
!:@2Adam/conv2d_26/bias/v
/:-@@2Adam/conv2d_27/kernel/v
!:@2Adam/conv2d_27/bias/v
(:&
?b?2Adam/dense_12/kernel/v
!:?2Adam/dense_12/bias/v
':%	?2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
?2?
,__inference_sequential_6_layer_call_fn_17158
,__inference_sequential_6_layer_call_fn_17620
,__inference_sequential_6_layer_call_fn_17649
,__inference_sequential_6_layer_call_fn_17462?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_16930?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_24_input?????????@@
?2?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17702
G__inference_sequential_6_layer_call_and_return_conditional_losses_17776
G__inference_sequential_6_layer_call_and_return_conditional_losses_17508
G__inference_sequential_6_layer_call_and_return_conditional_losses_17554?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_24_layer_call_fn_17785?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_17795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_36_layer_call_fn_17800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_36_layer_call_and_return_conditional_losses_17805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_25_layer_call_fn_17814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_17824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_37_layer_call_fn_17829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_37_layer_call_and_return_conditional_losses_17834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_12_layer_call_fn_16942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_16936?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_dropout_18_layer_call_fn_17839
*__inference_dropout_18_layer_call_fn_17844?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_18_layer_call_and_return_conditional_losses_17849
E__inference_dropout_18_layer_call_and_return_conditional_losses_17861?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_26_layer_call_fn_17870?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_17880?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_38_layer_call_fn_17885?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_38_layer_call_and_return_conditional_losses_17890?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_27_layer_call_fn_17899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_17909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_39_layer_call_fn_17914?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_39_layer_call_and_return_conditional_losses_17919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_13_layer_call_fn_16954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_16948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_dropout_19_layer_call_fn_17924
*__inference_dropout_19_layer_call_fn_17929?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_19_layer_call_and_return_conditional_losses_17934
E__inference_dropout_19_layer_call_and_return_conditional_losses_17946?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_flatten_6_layer_call_fn_17951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_6_layer_call_and_return_conditional_losses_17957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_12_layer_call_fn_17966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_12_layer_call_and_return_conditional_losses_17976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_40_layer_call_fn_17981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_40_layer_call_and_return_conditional_losses_17986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_20_layer_call_fn_17991
*__inference_dropout_20_layer_call_fn_17996?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_20_layer_call_and_return_conditional_losses_18001
E__inference_dropout_20_layer_call_and_return_conditional_losses_18013?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_13_layer_call_fn_18022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_13_layer_call_and_return_conditional_losses_18032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_activation_41_layer_call_fn_18037?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_41_layer_call_and_return_conditional_losses_18042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_17591conv2d_24_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_16930?#$56?@UVcd@?=
6?3
1?.
conv2d_24_input?????????@@
? "=?:
8
activation_41'?$
activation_41??????????
H__inference_activation_36_layer_call_and_return_conditional_losses_17805h7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@ 
? ?
-__inference_activation_36_layer_call_fn_17800[7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@ ?
H__inference_activation_37_layer_call_and_return_conditional_losses_17834h7?4
-?*
(?%
inputs?????????>> 
? "-?*
#? 
0?????????>> 
? ?
-__inference_activation_37_layer_call_fn_17829[7?4
-?*
(?%
inputs?????????>> 
? " ??????????>> ?
H__inference_activation_38_layer_call_and_return_conditional_losses_17890h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
-__inference_activation_38_layer_call_fn_17885[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
H__inference_activation_39_layer_call_and_return_conditional_losses_17919h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
-__inference_activation_39_layer_call_fn_17914[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
H__inference_activation_40_layer_call_and_return_conditional_losses_17986Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
-__inference_activation_40_layer_call_fn_17981M0?-
&?#
!?
inputs??????????
? "????????????
H__inference_activation_41_layer_call_and_return_conditional_losses_18042X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_activation_41_layer_call_fn_18037K/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_conv2d_24_layer_call_and_return_conditional_losses_17795l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@ 
? ?
)__inference_conv2d_24_layer_call_fn_17785_7?4
-?*
(?%
inputs?????????@@
? " ??????????@@ ?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_17824l#$7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????>> 
? ?
)__inference_conv2d_25_layer_call_fn_17814_#$7?4
-?*
(?%
inputs?????????@@ 
? " ??????????>> ?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_17880l567?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_26_layer_call_fn_17870_567?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_17909l?@7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_27_layer_call_fn_17899_?@7?4
-?*
(?%
inputs?????????@
? " ??????????@?
C__inference_dense_12_layer_call_and_return_conditional_losses_17976^UV0?-
&?#
!?
inputs??????????b
? "&?#
?
0??????????
? }
(__inference_dense_12_layer_call_fn_17966QUV0?-
&?#
!?
inputs??????????b
? "????????????
C__inference_dense_13_layer_call_and_return_conditional_losses_18032]cd0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_13_layer_call_fn_18022Pcd0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dropout_18_layer_call_and_return_conditional_losses_17849l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
E__inference_dropout_18_layer_call_and_return_conditional_losses_17861l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
*__inference_dropout_18_layer_call_fn_17839_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
*__inference_dropout_18_layer_call_fn_17844_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
E__inference_dropout_19_layer_call_and_return_conditional_losses_17934l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
E__inference_dropout_19_layer_call_and_return_conditional_losses_17946l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
*__inference_dropout_19_layer_call_fn_17924_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
*__inference_dropout_19_layer_call_fn_17929_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
E__inference_dropout_20_layer_call_and_return_conditional_losses_18001^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_20_layer_call_and_return_conditional_losses_18013^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_20_layer_call_fn_17991Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_20_layer_call_fn_17996Q4?1
*?'
!?
inputs??????????
p
? "????????????
D__inference_flatten_6_layer_call_and_return_conditional_losses_17957a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????b
? ?
)__inference_flatten_6_layer_call_fn_17951T7?4
-?*
(?%
inputs?????????@
? "???????????b?
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_16936?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_12_layer_call_fn_16942?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_16948?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_13_layer_call_fn_16954?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_sequential_6_layer_call_and_return_conditional_losses_17508#$56?@UVcdH?E
>?;
1?.
conv2d_24_input?????????@@
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17554#$56?@UVcdH?E
>?;
1?.
conv2d_24_input?????????@@
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17702v#$56?@UVcd??<
5?2
(?%
inputs?????????@@
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_6_layer_call_and_return_conditional_losses_17776v#$56?@UVcd??<
5?2
(?%
inputs?????????@@
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_6_layer_call_fn_17158r#$56?@UVcdH?E
>?;
1?.
conv2d_24_input?????????@@
p 

 
? "???????????
,__inference_sequential_6_layer_call_fn_17462r#$56?@UVcdH?E
>?;
1?.
conv2d_24_input?????????@@
p

 
? "???????????
,__inference_sequential_6_layer_call_fn_17620i#$56?@UVcd??<
5?2
(?%
inputs?????????@@
p 

 
? "???????????
,__inference_sequential_6_layer_call_fn_17649i#$56?@UVcd??<
5?2
(?%
inputs?????????@@
p

 
? "???????????
#__inference_signature_wrapper_17591?#$56?@UVcdS?P
? 
I?F
D
conv2d_24_input1?.
conv2d_24_input?????????@@"=?:
8
activation_41'?$
activation_41?????????