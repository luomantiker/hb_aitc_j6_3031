from enum import Enum
from typing import Tuple


class ColorConvert(Enum):
    """Enum for different types of color conversions.

    The color conversion type is used to convert the original
    color type of model input to the expected color type.
    """

    NULL = 0
    # same input type, input range changes from 0~255 to -128~127.
    RGB_TO_RGB_128 = 1
    BGR_TO_BGR_128 = 2
    YUV444_TO_YUV444_128 = 3
    GRAY_TO_GRAY_128 = 4
    # different input type, same input range.
    RGB_TO_BGR = 5
    BGR_TO_RGB = 6
    RGB_TO_YUV444 = 7
    BGR_TO_YUV444 = 8
    # different input type, input range changes from 0~255 to -128~127.
    RGB_TO_BGR_128 = 9
    BGR_TO_RGB_128 = 10
    RGB_TO_YUV444_128 = 11
    BGR_TO_YUV444_128 = 12
    RGB_TO_YUV_BT601_FULL_RANGE = 13
    RGB_TO_YUV_BT601_VIDEO_RANGE = 14
    BGR_TO_YUV_BT601_FULL_RANGE = 15
    BGR_TO_YUV_BT601_VIDEO_RANGE = 16

    @classmethod
    def get_convert_type(cls, original_type: str, expected_type: str) -> "ColorConvert":
        """Get the color conversion type from original type to expected type.

        Args:
            original_type: The original color type.
            expected_type: The expected color type.

        Returns:
            The corresponding color conversion enum value.

        Raises:
            ValueError: If the color conversion type is not supported.
        """
        original_type = original_type.upper()
        expected_type = expected_type.upper()
        if original_type == expected_type:
            return cls.NULL

        convert_type = f"{original_type}_TO_{expected_type}"
        if not hasattr(cls, convert_type):
            raise ValueError(
                f"The color conversion from {original_type} to "
                f"{expected_type} is not supported."
            )

        return getattr(cls, convert_type)

    @classmethod
    def split_color_convert(cls, color_convert: "ColorConvert") -> Tuple[str, str]:
        """Split the color conversion type into original type and expected type.

        Args:
            color_convert: The color conversion enum value.

        Returns:
            The original type and expected type.

        Raises:
            ValueError: If the color conversion type is not supported.
        """
        if color_convert not in cls.__members__.values():
            raise ValueError(f"Unsupported color conversion type: {color_convert}.")
        if color_convert != cls.NULL:
            return tuple(color_convert.name.split("_TO_"))

        return ("RGB", "RGB")
