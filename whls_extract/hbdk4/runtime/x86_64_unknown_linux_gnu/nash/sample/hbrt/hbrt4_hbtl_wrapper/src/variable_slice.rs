use core::ffi::c_char;
use core::mem::size_of;
use core::ptr;
use core::slice;
use hbrt4_header::{
    hbrt4MemspaceGetSize, hbrt4TypeGetArrayNumElements, hbrt4TypeGetElementType, hbrt4TypeGetTag,
    hbrt4TypeGetTensorDims, hbrt4TypeGetTensorElementType, hbrt4TypeGetTensorStrides,
    hbrt4VariableGetMemspace, hbrt4VariableGetOffsetInMemspace, hbrt4VariableGetType,
    Hbrt4Memspace, Hbrt4PtrdiffTArrayRef, Hbrt4Status, Hbrt4Type, Hbrt4TypeTag, Hbrt4Variable,
};
use hbrt4_log::error::LogAnyError;
use hbrt4_log::loge;
use ndarray::Array;
use ndarray::Dim;
use ndarray::IxDyn;
use ndarray::IxDynImpl;
use ndarray::ShapeBuilder;

use crate::HbtlVariableWrapper;
use crate::{hbrt_cpu_args_type_t, hbrt_element_type_t};

pub type HbtlTensorNdArray<A> = Array<A, Dim<IxDynImpl>>;
#[derive(Debug)]
pub struct VariableMessage<'a> {
    pub offset: usize,
    pub data_len: Option<usize>,
    pub dims: Option<&'a [isize]>,
    pub strides: Option<&'a [isize]>,
    pub data_type: hbrt_element_type_t,
    pub cpu_args_type: hbrt_cpu_args_type_t,
    pub is_dynamic_shape: bool,
}

impl<'a> TryFrom<&Hbrt4Variable> for VariableMessage<'a> {
    type Error = LogAnyError;

    fn try_from(var: &Hbrt4Variable) -> Result<Self, Self::Error> {
        let mut var_type = Hbrt4Type::null_obj();
        let mut type_tag = Hbrt4TypeTag::Unknown;
        let mut type_dims = Hbrt4PtrdiffTArrayRef::default();
        let mut type_strides = Hbrt4PtrdiffTArrayRef::default();
        let mut data_type = hbrt_element_type_t::ELEMENT_TYPE_UNKNOWN;
        let cpu_args_type;
        let data_len;
        unsafe {
            hbrt4VariableGetType(*var, &mut var_type);
            hbrt4TypeGetTag(var_type, &mut type_tag);
            hbrt4TypeGetTensorDims(var_type, &mut type_dims);
            hbrt4TypeGetTensorStrides(var_type, &mut type_strides);
            (data_len, cpu_args_type) = match type_tag {
                Hbrt4TypeTag::Si64 => (Some(size_of::<i64>()), hbrt_cpu_args_type_t::INT64),
                Hbrt4TypeTag::Bool => (Some(size_of::<i8>()), hbrt_cpu_args_type_t::BOOL),
                Hbrt4TypeTag::F32 => (Some(size_of::<f32>()), hbrt_cpu_args_type_t::FLOAT),
                Hbrt4TypeTag::F64 => (Some(size_of::<f64>()), hbrt_cpu_args_type_t::DOUBLE),
                Hbrt4TypeTag::String => {
                    // TODO(shanpengxiang): record string len
                    (Some(1_usize), hbrt_cpu_args_type_t::STRING)
                }
                Hbrt4TypeTag::Tensor => {
                    data_type = get_element_type(var_type)?;
                    (None, hbrt_cpu_args_type_t::TENSOR)
                }
                Hbrt4TypeTag::Array => Self::get_element_cpu_args_type(var_type)?,
                _ => {
                    return Err(loge!(/NotImpl, "not supported type"));
                }
            };
        }

        let mut memspace = Hbrt4Memspace::null_obj();
        let mut offset = 0;
        let mut size = 0;
        unsafe {
            hbrt4VariableGetMemspace(*var, &mut memspace);
            hbrt4VariableGetOffsetInMemspace(*var, &mut offset);
            hbrt4MemspaceGetSize(memspace, &mut size);
        }
        if type_tag == Hbrt4TypeTag::Tensor {
            let dims_data = type_dims.data;
            let strides_data = type_strides.data;

            let dims = unsafe { slice::from_raw_parts(dims_data, type_dims.len) };
            let strides = unsafe { slice::from_raw_parts(strides_data, type_strides.len) };
            let is_dynamic = dims.contains(&isize::MIN);
            if type_dims.len > 0 && strides[0] != isize::MIN && !is_dynamic {
                // scalar or dynamic dim/stride don't need the check
                if size < (dims[0] * strides[0]).try_into().unwrap() {
                    return Err(loge!(/Int, "invalid dim and stride"));
                }
            }
            Ok(VariableMessage {
                offset,
                data_len: None,
                dims: Some(dims),
                strides: Some(strides),
                data_type,
                cpu_args_type,
                is_dynamic_shape: is_dynamic,
            })
        } else {
            Ok(VariableMessage {
                offset,
                data_len,
                dims: None,
                strides: None,
                data_type,
                cpu_args_type,
                is_dynamic_shape: false,
            })
        }
    }
}

impl<'a> VariableMessage<'a> {
    fn get_element_cpu_args_type(
        var_type: Hbrt4Type,
    ) -> Result<(Option<usize>, hbrt_cpu_args_type_t), LogAnyError> {
        let mut elem_type = Hbrt4Type::null_obj();
        let mut elem_type_tag = Hbrt4TypeTag::Unknown;
        unsafe {
            let mut num = 0_usize;
            hbrt4TypeGetElementType(var_type, &mut elem_type);
            hbrt4TypeGetArrayNumElements(var_type, &mut num);
            hbrt4TypeGetTag(elem_type, &mut elem_type_tag);
            let (data_len, cpu_args_type) = match elem_type_tag {
                Hbrt4TypeTag::Si64 => (
                    Some(num * size_of::<i64>()),
                    hbrt_cpu_args_type_t::I64_ARRAY,
                ),
                Hbrt4TypeTag::F32 => (
                    Some(num * size_of::<f32>()),
                    hbrt_cpu_args_type_t::F32_ARRAY,
                ),
                Hbrt4TypeTag::F64 => (
                    Some(num * size_of::<f64>()),
                    hbrt_cpu_args_type_t::F64_ARRAY,
                ),
                _ => {
                    return Err(loge!(/NotImpl, "not supported tensor array type"));
                }
            };
            Ok((data_len, cpu_args_type))
        }
    }

    pub fn new(var: &Hbrt4Variable) -> Result<VariableMessage<'a>, LogAnyError> {
        unsafe {
            let mut var_type = Hbrt4Type::null_obj();
            let mut type_tag = Hbrt4TypeTag::Unknown;
            let mut memspace = Hbrt4Memspace::null_obj();

            hbrt4VariableGetType(*var, &mut var_type);
            hbrt4TypeGetTag(var_type, &mut type_tag);

            if hbrt4VariableGetMemspace(*var, &mut memspace) == Hbrt4Status::Ok {
                Ok(var.try_into()?)
            } else {
                Err(loge!(/NotImpl, "not memspace for variable"))
            }
        }
    }
    pub fn convert_variable_to_hbtl_variable(
        &self,
        var_slice: &mut [c_char],
        dynamic_shape_vec: Option<&mut Vec<i64>>,
    ) -> HbtlVariableWrapper {
        let shapes_ptr = if self.is_dynamic_shape {
            match dynamic_shape_vec {
                Some(v) => v.as_mut_slice().as_mut_ptr(),
                None => ptr::null_mut(),
            }
        } else {
            ptr::null_mut()
        };
        let dims_ptr = if self.is_dynamic_shape {
            shapes_ptr
        } else {
            match self.dims {
                Some(t) => t.as_ptr().cast::<i64>(),
                None => ptr::null_mut(),
            }
        };
        HbtlVariableWrapper {
            data: var_slice.as_ptr().cast_mut(),
            dataLen: match self.data_len {
                Some(t) => t,
                None => var_slice.len(),
            },
            stride: match self.strides {
                Some(t) => t.as_ptr().cast::<i64>(),
                None => ptr::null_mut(),
            },
            size: dims_ptr,
            rank: match self.dims {
                Some(t) => t.len(),
                None => 0,
            },
            hbrtDataType: self.data_type,
            hbrtCpuArgsType: self.cpu_args_type,
            isDynamicVariable: self.is_dynamic_shape,
            dynamicSize: shapes_ptr,
        }
    }

    // Note: Only used for pyramid and resizer
    pub fn convert_dynamic_variable_to_hbtl_variable(
        &self,
        var_slice: &mut [c_char],
        strides: &Hbrt4PtrdiffTArrayRef,
        dims: &Hbrt4PtrdiffTArrayRef,
    ) -> HbtlVariableWrapper {
        HbtlVariableWrapper {
            data: var_slice.as_ptr().cast_mut(),
            dataLen: var_slice.len(),
            stride: strides.data.cast::<i64>(),
            size: dims.data.cast::<i64>(),
            rank: dims.len,
            hbrtDataType: self.data_type,
            hbrtCpuArgsType: self.cpu_args_type,
            isDynamicVariable: self.is_dynamic_shape,
            dynamicSize: ptr::null_mut(),
        }
    }

    #[allow(clippy::cast_sign_loss)]
    pub fn convert_to_ndarray<T: Clone>(
        &self,
        var_slice: &[T],
    ) -> Result<HbtlTensorNdArray<T>, LogAnyError> {
        let dims_vec: Vec<_> = self.dims.unwrap().iter().map(|v| *v as usize).collect();
        let elem_type = size_of::<T>();
        let strides_vec: Vec<_> = self
            .strides
            .unwrap()
            .iter()
            .map(|v| *v as usize / elem_type)
            .collect();
        let var_dims = IxDyn(&dims_vec);
        let var_strides = IxDyn(&strides_vec);
        let var_array = Array::from_shape_vec(var_dims.strides(var_strides), var_slice.to_vec())
            .map_err(|e| loge!(/Int, "convert ndarray error [ {} ]", e.to_string()))?;
        Ok(var_array)
    }
}

pub fn get_element_type(var_type: Hbrt4Type) -> Result<hbrt_element_type_t, LogAnyError> {
    let mut elem_type = Hbrt4Type::null_obj();
    let mut elem_type_tag = Hbrt4TypeTag::Unknown;
    unsafe {
        hbrt4TypeGetTensorElementType(var_type, &mut elem_type);
        hbrt4TypeGetTag(elem_type, &mut elem_type_tag);
        let res = match elem_type_tag {
            Hbrt4TypeTag::Si8 => hbrt_element_type_t::ELEMENT_TYPE_INT8,
            Hbrt4TypeTag::Si16 => hbrt_element_type_t::ELEMENT_TYPE_INT16,
            Hbrt4TypeTag::Si32 => hbrt_element_type_t::ELEMENT_TYPE_INT32,
            Hbrt4TypeTag::Si64 => hbrt_element_type_t::ELEMENT_TYPE_INT64,
            Hbrt4TypeTag::Ui8 => hbrt_element_type_t::ELEMENT_TYPE_UINT8,
            Hbrt4TypeTag::Ui16 => hbrt_element_type_t::ELEMENT_TYPE_UINT16,
            Hbrt4TypeTag::Ui32 => hbrt_element_type_t::ELEMENT_TYPE_UINT32,
            Hbrt4TypeTag::Ui64 => hbrt_element_type_t::ELEMENT_TYPE_UINT64,
            Hbrt4TypeTag::F16 => hbrt_element_type_t::ELEMENT_TYPE_FLOAT16,
            Hbrt4TypeTag::F32 => hbrt_element_type_t::ELEMENT_TYPE_FLOAT32,
            Hbrt4TypeTag::F64 => hbrt_element_type_t::ELEMENT_TYPE_FLOAT64,
            Hbrt4TypeTag::Bool => hbrt_element_type_t::ELEMENT_TYPE_BOOL8,
            _ => {
                return Err(loge!(/NotImpl, "not supported array type"));
            }
        };
        Ok(res)
    }
}

pub fn get_element_size(elem_type: hbrt_element_type_t) -> Result<usize, LogAnyError> {
    let res = match elem_type {
        hbrt_element_type_t::ELEMENT_TYPE_INT8 => size_of::<i8>(),
        hbrt_element_type_t::ELEMENT_TYPE_INT16 => size_of::<i16>(),
        hbrt_element_type_t::ELEMENT_TYPE_INT32 => size_of::<i32>(),
        hbrt_element_type_t::ELEMENT_TYPE_INT64 => size_of::<i64>(),
        hbrt_element_type_t::ELEMENT_TYPE_UINT8 => size_of::<u8>(),
        // FIXME(wuruidong): Due to there is none size_of::<f16>, so `u16` is used for `FLOAT16`.
        hbrt_element_type_t::ELEMENT_TYPE_UINT16 | hbrt_element_type_t::ELEMENT_TYPE_FLOAT16 => {
            size_of::<u16>()
        }
        hbrt_element_type_t::ELEMENT_TYPE_UINT32 => size_of::<u32>(),
        hbrt_element_type_t::ELEMENT_TYPE_UINT64 => size_of::<u64>(),
        hbrt_element_type_t::ELEMENT_TYPE_FLOAT32 => size_of::<f32>(),
        hbrt_element_type_t::ELEMENT_TYPE_FLOAT64 => size_of::<f64>(),
        hbrt_element_type_t::ELEMENT_TYPE_BOOL8 => size_of::<bool>(),
        _ => {
            return Err(loge!(/NotImpl, "not supported element type"));
        }
    };
    Ok(res)
}

pub fn convert_tuple_variable_hbtl_variable(
    vec: &mut [HbtlVariableWrapper],
) -> HbtlVariableWrapper {
    HbtlVariableWrapper {
        data: vec.as_ptr() as *mut ::core::ffi::c_char,
        dataLen: vec.len(),
        stride: ptr::null_mut(),
        size: ptr::null_mut(),
        rank: 0,
        hbrtDataType: hbrt_element_type_t::ELEMENT_TYPE_UNKNOWN,
        hbrtCpuArgsType: hbrt_cpu_args_type_t::TENSOR_ARRAY,
        isDynamicVariable: false,
        dynamicSize: ptr::null_mut(),
    }
}
