#!/usr/bin/python3
"""Script for creating basenet with one Bottleneck LSTM layer after conv 13.
"""
import torch
import torch.nn as nn

from hbdk4.compiler.eval_kit import trace


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    Arguments:
        in_channels : number of channels of input
        out_channels : number of channels of output
        kernel_size : kernel size for depthwise convolution
        stride : stride for depthwise convolution
        padding : padding for depthwise convolution
    Returns:
        object of class torch.nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(in_channels),
            kernel_size=kernel_size,
            groups=int(in_channels),
            stride=stride,
            padding=padding,
        ),
        nn.ReLU6(),
        nn.Conv2d(
            in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=1
        ),
    )


def conv_bn(inp, oup, stride):
    """3x3 conv with batchnorm and relu
    Arguments:
        inp : number of channels of input
        oup : number of channels of output
        stride : stride for depthwise convolution
    Returns:
        object of class torch.nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(int(inp), int(oup), 3, stride, 1, bias=False),
        nn.BatchNorm2d(int(oup)),
        nn.ReLU6(inplace=True),
    )


def conv_dw(inp, oup, stride):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d having batchnorm and relu layers in between.
    Here kernel size is fixed at 3.
    Arguments:
        inp : number of channels of input
        oup : number of channels of output
        stride : stride for depthwise convolution
    Returns:
        object of class torch.nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(int(inp), int(inp), 3, stride, 1, groups=int(inp), bias=False),
        nn.BatchNorm2d(int(inp)),
        nn.ReLU6(inplace=True),
        nn.Conv2d(int(inp), int(oup), 1, 1, 0, bias=False),
        nn.BatchNorm2d(int(oup)),
        nn.ReLU6(inplace=True),
    )


class BottleneckLSTMCell(nn.Module):
    """Creates a LSTM layer cell
    Arguments:
        input_channels : variable used to contain value of number of channels in input
        hidden_channels : variable used to contain value of number of channels in the hidden state of LSTM cell
    """

    def __init__(self, input_channels, hidden_channels):
        super(BottleneckLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_features = 4
        self.W = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.input_channels,
            kernel_size=3,
            groups=self.input_channels,
            stride=1,
            padding=1,
        )
        self.Wy = nn.Conv2d(
            int(self.input_channels + self.hidden_channels),
            self.hidden_channels,
            kernel_size=1,
        )
        self.Wi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            3,
            1,
            1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.Wbi = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False
        )
        self.Wbf = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False
        )
        self.Wbc = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False
        )
        self.Wbo = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False
        )
        self.relu = nn.ReLU6()

    def forward(
        self, x, h, c
    ):  # implemented as mentioned in paper here the only difference is  Wbi, Wbf, Wbc & Wbo are commuted all together in paper
        """
        Arguments:
            x : input tensor
            h : hidden state tensor
            c : cell state tensor
        Returns:
            output tensor after LSTM cell
        """
        x = self.W(x)
        y = torch.cat((x, h), 1)  # concatenate input and hidden layers
        i = self.Wy(y)  # reduce to hidden layer size
        b = self.Wi(i)  # depth wise 3*3
        ci = torch.sigmoid(self.Wbi(b))
        cf = torch.sigmoid(self.Wbf(b))
        cc = cf * c + ci * self.relu(self.Wbc(b))
        co = torch.sigmoid(self.Wbo(b))
        ch = co * self.relu(cc)
        return ch, cc


class BottleneckLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, batch_size):
        """Creates Bottleneck LSTM layer
        Arguments:
            input_channels : variable having value of number of channels of input to this layer
            hidden_channels : variable having value of number of channels of hidden state of this layer
            batch_size : an int variable having value of batch_size of the input
        Returns:
            Output tensor of LSTM layer
        """
        super(BottleneckLSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels)
        self.batch = batch_size

    # self.register_buffer('hidden_state', torch.zeros(batch_size, self.hidden_channels, height, width))
    # self.register_buffer('cell_state', torch.zeros(batch_size, self.hidden_channels, height, width))

    def forward(self, input, h, c):
        hidden = []
        cell = []
        for b in range(self.batch):
            h, c = self.cell(input[b].unsqueeze(0), h, c)
            hidden.append(h)
            cell.append(c)

        return torch.cat(hidden, 0), torch.cat(cell, 0)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024, alpha=1):
        """torch.nn.module for mobilenetv1 upto conv12
        Arguments:
            num_classes : an int variable having value of total number of classes
            alpha : a float used as width multiplier for channels of model
        """
        super(MobileNetV1, self).__init__()
        # upto conv 12
        self.model = nn.Sequential(
            conv_bn(3, 32 * alpha, 2),
            conv_dw(32 * alpha, 64 * alpha, 1),
            conv_dw(64 * alpha, 128 * alpha, 2),
            conv_dw(128 * alpha, 128 * alpha, 1),
            conv_dw(128 * alpha, 256 * alpha, 2),
            conv_dw(256 * alpha, 256 * alpha, 1),
            conv_dw(256 * alpha, 512 * alpha, 2),
            conv_dw(512 * alpha, 512 * alpha, 1),
            conv_dw(512 * alpha, 512 * alpha, 1),
            conv_dw(512 * alpha, 512 * alpha, 1),
            conv_dw(512 * alpha, 512 * alpha, 1),
            conv_dw(512 * alpha, 512 * alpha, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SSD(nn.Module):
    def __init__(self, num_classes, batch_size, alpha=1, is_test=False, config=None):
        """
        Arguments:
            num_classes : an int variable having value of total number of classes
            batch_size : an int variable having value of batch size
            alpha : a float used as width multiplier for channels of model
            is_Test : a bool used to make model ready for testing
            config : a dict containing all the configuration parameters
        """
        super(SSD, self).__init__()
        # Decoder
        self.num_classes = num_classes
        self.conv13 = conv_dw(
            512 * alpha, 1024 * alpha, 2
        )  # not using conv14 as mentioned in paper
        self.bottleneck_lstm1 = BottleneckLSTM(
            input_channels=1024 * alpha,
            hidden_channels=256 * alpha,
            batch_size=batch_size,
        )
        self.fmaps_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * alpha),
                out_channels=int(128 * alpha),
                kernel_size=1,
            ),
            nn.ReLU6(inplace=True),
            SeperableConv2d(
                in_channels=128 * alpha,
                out_channels=256 * alpha,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.fmaps_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(256 * alpha),
                out_channels=int(64 * alpha),
                kernel_size=1,
            ),
            nn.ReLU6(inplace=True),
            SeperableConv2d(
                in_channels=64 * alpha,
                out_channels=128 * alpha,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.fmaps_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * alpha),
                out_channels=int(64 * alpha),
                kernel_size=1,
            ),
            nn.ReLU6(inplace=True),
            SeperableConv2d(
                in_channels=64 * alpha,
                out_channels=128 * alpha,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.fmaps_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(128 * alpha),
                out_channels=int(32 * alpha),
                kernel_size=1,
            ),
            nn.ReLU6(inplace=True),
            SeperableConv2d(
                in_channels=32 * alpha,
                out_channels=64 * alpha,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        self.regression_headers = nn.ModuleList(
            [
                SeperableConv2d(
                    in_channels=512 * alpha,
                    out_channels=6 * 4,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=256 * alpha,
                    out_channels=6 * 4,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=256 * alpha,
                    out_channels=6 * 4,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=128 * alpha,
                    out_channels=6 * 4,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=128 * alpha,
                    out_channels=6 * 4,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=int(64 * alpha), out_channels=6 * 4, kernel_size=1
                ),
            ]
        )

        self.classification_headers = nn.ModuleList(
            [
                SeperableConv2d(
                    in_channels=512 * alpha,
                    out_channels=6 * num_classes,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=256 * alpha,
                    out_channels=6 * num_classes,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=256 * alpha,
                    out_channels=6 * num_classes,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=128 * alpha,
                    out_channels=6 * num_classes,
                    kernel_size=3,
                    padding=1,
                ),
                SeperableConv2d(
                    in_channels=128 * alpha,
                    out_channels=6 * num_classes,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=int(64 * alpha),
                    out_channels=6 * num_classes,
                    kernel_size=1,
                ),
            ]
        )

    def compute_header(self, i, x):  # ssd method to calculate headers
        """
        Arguments:
            i : an int used to use particular classification and regression layer
            x : a tensor used as input to layers
        Returns:
            locations and confidences of the predictions
        """
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def forward(self, x, hidden, cell):
        """
        Arguments:
            x : a tensor which is used as input for the model
        Returns:
            confidences and locations of predictions made by model during training
            or
            confidences and boxes of predictions made by model during testing
        """

        confidences = []
        locations = []
        header_index = 0
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        x = self.conv13(x)
        hidden, cell = self.bottleneck_lstm1(x, hidden, cell)
        x = hidden

        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        x = self.fmaps_1(x)
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        x = self.fmaps_2(x)
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        x = self.fmaps_3(x)
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)

        x = self.fmaps_4(x)
        confidence, location = self.compute_header(header_index, x)
        header_index += 1
        confidences.append(confidence)
        locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        return confidences, locations, hidden[-2:], cell[-2:]


class MobileVOD(nn.Module):
    """
    Module to join encoder and decoder of predictor model
    """

    def __init__(self, pred_enc, pred_dec):
        """
        Arguments:
            pred_enc : an object of MobilenetV1 class
            pred_dec : an object of SSD class
        """
        super(MobileVOD, self).__init__()
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec

    def forward(self, x, hidden, cell):
        x = self.pred_encoder(torch.permute(x, [0, 3, 1, 2]))
        hidden = torch.permute(hidden, [0, 3, 1, 2])
        cell = torch.permute(cell, [0, 3, 1, 2])
        confidences, locations, hidden, cell = self.pred_decoder(x, hidden, cell)
        return (
            confidences,
            locations,
            torch.permute(hidden, [0, 2, 3, 1]),
            torch.permute(cell, [0, 2, 3, 1]),
        )


def retrieve_bottleneck_lstm(batch, pretrained=False, post_process=False):
    height = 320
    width = 320
    assert post_process is False
    ei = (
        torch.rand(batch, height, width, 3),
        torch.rand(1, height // 32, width // 32, 256),
        torch.rand(1, height // 32, width // 32, 256),
    )
    return (
        trace(MobileVOD(MobileNetV1(), SSD(30, batch)), ei, splat=not pretrained),
        ei,
    )
