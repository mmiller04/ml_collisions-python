import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)

class UnetDownBlock(nn.Module):
   
    def __init__(self, inplanes, planes, predownsample_block):
        
        super(UnetDownBlock, self).__init__()
        
        self.predownsample_block = predownsample_block
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
    def forward(self, x):
        
        x = self.predownsample_block(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x
    
class UnetUpBlock(nn.Module):
   
    def __init__(self, inplanes, planes, postupsample_block=None):
        
        super(UnetUpBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        if postupsample_block is None: 
            
            self.postupsample_block = nn.ConvTranspose2d(in_channels=planes,
                                                         out_channels=planes//2,
                                                         kernel_size=2,
                                                         stride=2)
            
        else:
            
            self.postupsample_block = postupsample_block
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.postupsample_block(x)
        
        return x
    
    
class Unet(nn.Module):
    
    def __init__(self):
        
        super(Unet, self).__init__()
        
        self.predownsample_block = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.identity_block = nn.Sequential()
        
        self.block1 = UnetDownBlock(
                                    predownsample_block=self.identity_block,
                                    inplanes=2, planes=64
                                    )
        
        self.block2_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=64, planes=128
                                         )
        
        self.block3_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=128, planes=256
                                         )

        self.block4_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=256, planes=512
                                         )
        
        self.block5_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=512, planes=1024
                                         )
        
        self.block1_up = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                                  kernel_size=2, stride=2)
        
        self.block2_up = UnetUpBlock(
                                     inplanes=1024, planes=512
                                     )
        
        self.block3_up = UnetUpBlock(
                                     inplanes=512, planes=256
                                     )
        
        self.block4_up = UnetUpBlock(
                                     inplanes=256, planes=128
                                     )
        
        self.block5 = UnetUpBlock(
                                  inplanes=128, planes=64,
                                  postupsample_block=self.identity_block
                                  )
        
        self.logit_conv = nn.Conv2d(
                                    in_channels=64, out_channels=1, kernel_size=1,
                                    )
        
        
    def forward(self, x):
        
        features_1s_down = self.block1(x)
        features_2s_down = self.block2_down(features_1s_down)
        features_4s_down = self.block3_down(features_2s_down)
        features_8s_down = self.block4_down(features_4s_down)
        
        features_16s = self.block5_down(features_8s_down)
        
        features_8s_up = self.block1_up(features_16s)
        features_8s_up = torch.cat([features_8s_down, features_8s_up],dim=1)
        
        features_4s_up = self.block2_up(features_8s_up)
        features_4s_up = torch.cat([features_4s_down, features_4s_up],dim=1)
        
        features_2s_up = self.block3_up(features_4s_up)
        features_2s_up = torch.cat([features_2s_down, features_2s_up],dim=1)
        
        features_1s_up = self.block4_up(features_2s_up)
        features_1s_up = torch.cat([features_1s_down, features_1s_up],dim=1)
        
        features_final = self.block5(features_1s_up)
        
        logits = self.logit_conv(features_final)
       
        return logits

class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 2
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

  
class VGG16(nn.Module):
    
    def __init__(self, n_layers, usegpu=True):
        super(VGG16,self).__init__()
        
        self.cnn = vgg16(pretrained=False)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        self.cnn = nn.Sequential(*list(self.cnn.children())[:n_layers])
        
    def __get_outputs(self,x):
        
        outputs = []
        for i, layer in enumerate(self.cnn.children()):
            x = layer(x)
            outputs.append(x)
            
        return outputs
    
    def forward(self,x):
        outputs = self.__get_outputs(x)
        
        return outputs[-1]
   
    
class SkipVGG16(nn.Module):
    
    def __init__(self, usegpu=True):
        super(SkipVGG16, self).__init__()
        
        self.outputs = [3,8]
        self.n_filters = [64,128]
        
        self.model = VGG16(n_layers=16, usegpu=usegpu)
        
    def forward(self,x):
        
        out = []
        for i, layer in enumerate(list(self.model.children())[0]):
            x = layer(x)
            if i in self.outputs:
                out.append(x)
        out.append(x)
        
        return out
    
    
class ReNet(nn.Module):
    
    def __init__(self, n_input, n_units, patch_size=(1, 1), usegpu=True):
        super(ReNet, self).__init__()
        
        self.usegpu=usegpu
        
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])
        
        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1
        
        self.tiling = False if ((self.patch_size_height == 1) and (
            self.patch_size_width == 1)) else True
                
        rnn_hor_n_inputs = n_input * self.patch_size_height * \
            self.patch_size_width
            
        self.rnn_hor = nn.GRU(rnn_hor_n_inputs, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)
        
        self.rnn_ver = nn.GRU(n_units * 2, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)
        
    def __tile(self,x):

        if (x.size(2) % self.patch_size_height) == 0:
            n_height_padding = 0
        else:
            n_height_padding = self.patch_size_height - \
                x.size(2) % self.patch_size_height
        if (x.size(3) % self.patch_size_width) == 0:
            n_width_padding = 0
        else:
            n_width_padding = self.patch_size_width - \
                x.size(3) % self.patch_size_width

        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding

        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding

        x = F.pad(x, (n_left_padding, n_right_padding,
                      n_top_padding, n_bottom_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0

        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width

        x = x.view(b, n_filters, new_height, self.patch_size_height,
                   new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height *
                   self.patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x
                
    def __swap_hw(self, x):

        # x : b, nf, h, w
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous()
        #  x : b, nf, w, h

        return x
    
    def rnn_forward(self, x, hor_or_ver):

        # x : b, nf, h, w
        assert hor_or_ver in ['hor', 'ver']

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        b, n_height, n_width, n_filters = x.size()
        # x : b, h, w, nf

        x = x.view(b * n_height, n_width, n_filters)
        # x : b * h, w, nf
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        elif hor_or_ver == 'ver':
            x, _ = self.rnn_ver(x)
            
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)
        # x : b, h, w, nf

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        # x : b, nf, h, w

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        return x
    
    def forward(self, x):

        # x : b, nf, h, w
        if self.tiling:
            x = self.__tile(x)

        x = self.rnn_forward(x, 'hor')
        x = self.rnn_forward(x, 'ver')

        return x
        

class ReSeg(nn.Module):
    
    def __init__(self, usegpu=True):
        super(ReSeg, self).__init__()
        
        self.cnn = SkipVGG16(usegpu=usegpu)
        
        self.renet1 = ReNet(256, 100, usegpu=usegpu)
        self.renet2 = ReNet(200, 100, usegpu=usegpu)
        
        self.upsampling1 = nn.ConvTranspose2d(200, 100,
                                              kernel_size=2,stride=2)
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(100 + self.cnn.n_filters[1], 100,
                                              kernel_size=2,stride=2)
        self.relu2 = nn.ReLU()
        
        self.final = nn.Conv2d(
                                in_channels = 100 + self.cnn.n_filters[0], 
                                out_channels = 1,
                                kernel_size=1,stride=1)
        
    def forward(self, x):
        
        first_skip, second_skip, x_enc = self.cnn(x)
        x_enc = self.renet1(x_enc)
        x_enc = self.renet2(x_enc)
        x_dec = self.relu1(self.upsampling1(x_enc))
        x_dec = torch.cat((x_dec, second_skip), dim=1)
        x_dec = self.relu2(self.upsampling2(x_dec))
        x_dec = torch.cat((x_dec, first_skip), dim=1)
        x_out = self.final(x_dec)
        
        return x_out

class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 2
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

  
class VGG16(nn.Module):
    
    def __init__(self, n_layers, usegpu=True):
        super(VGG16,self).__init__()
        
        self.cnn = vgg16(pretrained=False)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])
        self.cnn = nn.Sequential(*list(self.cnn.children())[:n_layers])
        
    def __get_outputs(self,x):
        
        outputs = []
        for i, layer in enumerate(self.cnn.children()):
            x = layer(x)
            outputs.append(x)
            
        return outputs
    
    def forward(self,x):
        outputs = self.__get_outputs(x)
        
        return outputs[-1]
  
    
class ReNet(nn.Module):
    
    def __init__(self, n_input, n_units, patch_size=(1, 1), usegpu=True):
        super(ReNet, self).__init__()
        
        self.usegpu=usegpu
        
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])
        
        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1
        
        self.tiling = False if ((self.patch_size_height == 1) and (
            self.patch_size_width == 1)) else True
                
        rnn_hor_n_inputs = n_input * self.patch_size_height * \
            self.patch_size_width
            
        self.rnn_hor = nn.GRU(rnn_hor_n_inputs, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)
        
        self.rnn_ver = nn.GRU(n_units * 2, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)
        
    def __tile(self,x):

        if (x.size(2) % self.patch_size_height) == 0:
            n_height_padding = 0
        else:
            n_height_padding = self.patch_size_height - \
                x.size(2) % self.patch_size_height
        if (x.size(3) % self.patch_size_width) == 0:
            n_width_padding = 0
        else:
            n_width_padding = self.patch_size_width - \
                x.size(3) % self.patch_size_width

        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding

        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding

        x = F.pad(x, (n_left_padding, n_right_padding,
                      n_top_padding, n_bottom_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0

        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width

        x = x.view(b, n_filters, new_height, self.patch_size_height,
                   new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height *
                   self.patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x
                
    def __swap_hw(self, x):

        # x : b, nf, h, w
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous()
        #  x : b, nf, w, h

        return x
    
    def rnn_forward(self, x, hor_or_ver):

        # x : b, nf, h, w
        assert hor_or_ver in ['hor', 'ver']

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        b, n_height, n_width, n_filters = x.size()
        # x : b, h, w, nf

        x = x.view(b * n_height, n_width, n_filters)
        # x : b * h, w, nf
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        elif hor_or_ver == 'ver':
            x, _ = self.rnn_ver(x)
            
        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)
        # x : b, h, w, nf

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        # x : b, nf, h, w

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        return x
    
    def forward(self, x):

        # x : b, nf, h, w
        if self.tiling:
            x = self.__tile(x)

        x = self.rnn_forward(x, 'hor')
        x = self.rnn_forward(x, 'ver')

        return x


class ConvGRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, kernel_size, usegpu=True):
        super(ConvGRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.usegpu = usegpu
        
        _n_inputs = self.input_size + self.hidden_size
        self.conv_gates = nn.Conv2d(_n_inputs,
                                    2 * self.hidden_size,
                                    self.kernel_size,
                                    padding=self.kernel_size // 2)

        self.conv_ct = nn.Conv2d(_n_inputs, self.hidden_size,
                                 self.kernel_size,
                                 padding=self.kernel_size // 2)
        
    def forward(self, x, hidden):
        
        batch_size, _, height, width = x.size()
        
        if hidden is None:
            size_h = [batch_size, self.hidden_size, height, width]
            hidden = Variable(torch.zeros(size_h))

            if self.usegpu:
                hidden = hidden.cuda()
                
        c1 = self.conv_gates(torch.cat((x, hidden), dim=1))
        rt, ut = c1.chunk(2, 1)

        reset_gate = torch.sigmoid(rt)
        update_gate = torch.sigmoid(ut)

        gated_hidden = torch.mul(reset_gate, hidden)

        ct = torch.tanh(self.conv_ct(torch.cat((x, gated_hidden), dim=1)))

        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct

        return next_h
    

class RecurrentHourglass(nn.Module):
    
    def __init__(self, input_n_filters, hidden_n_filters, kernel_size,
                 n_levels, embedding_size, usegpu=True):
        super(RecurrentHourglass, self).__init__()
            
        assert n_levels >= 1
    
        self.input_n_filters = input_n_filters
        self.hidden_n_filters = hidden_n_filters
        self.kernel_size = kernel_size
        self.n_levels = n_levels
        self.embedding_size = embedding_size
        self.usegpu = usegpu
        
        self.convgru_cell = ConvGRUCell(self.hidden_n_filters,
                                        self.hidden_n_filters,
                                        self.kernel_size,
                                        self.usegpu)
        
        self.__generate_pre_post_convs()
        
    def __generate_pre_post_convs(self):
        
        def __get_conv(input_n_filters, output_n_filters):
            return nn.Conv2d(input_n_filters, output_n_filters,
                             self.kernel_size,
                             padding=self.kernel_size // 2)
            
        self.pre_conv_layers = [__get_conv(self.input_n_filters,
                                           self.hidden_n_filters), ]
    
        for _ in range(self.n_levels - 1):
            self.pre_conv_layers.append(__get_conv(self.hidden_n_filters,
                                                   self.hidden_n_filters))
        self.pre_conv_layers = nn.ModuleList(self.pre_conv_layers)
    
        self.post_conv_layers = [__get_conv(self.hidden_n_filters,
                                            self.embedding_size), ]
        for _ in range(self.n_levels - 1):
            self.post_conv_layers.append(__get_conv(self.hidden_n_filters,
                                                    self.hidden_n_filters))
        self.post_conv_layers = nn.ModuleList(self.post_conv_layers)
    
    def forward_encoding(self, x):
        
        convgru_outputs = []
        hidden = None
        for i in range(self.n_levels):
            x = F.relu(self.pre_conv_layers[i](x))
            hidden = self.convgru_cell(x, hidden)
            convgru_outputs.append(hidden)
            
        return convgru_outputs
    
    def forward_decoding(self, convgru_outputs):
        
        _last_conv_layer = self.post_conv_layers[self.n_levels - 1]
        _last_output = convgru_outputs[self.n_levels - 1]
        
        post_feature_map = F.relu(_last_conv_layer(_last_output))
        for i in range(self.n_levels - 1)[::-1]:
            post_feature_map += convgru_outputs[i]
            post_feature_map = self.post_conv_layers[i](post_feature_map)
            post_feature_map = F.relu(post_feature_map)
            
        return post_feature_map
    
    def forward(self, x):
        
        x = self.forward_encoding(x)
        x = self.forward_decoding(x)
        
        return x
    

class StackedRecurrentHourglass(nn.Module):
    
    def __init__(self, usegpu=True):
        super(StackedRecurrentHourglass, self).__init__()
        
        self.usegpu = usegpu
        
        self.base_cnn = self.__generate_base_cnn()
        
        self.enc_stacked_hourglass = self.__generate_enc_stacked_hg(64,3)
        
        self.stacked_renet = self.__generate_stacked_renet(64,2)
        
        self.decoder = self.__generate_decoder(64)
        
    def __generate_base_cnn(self):
        
        base_cnn = VGG16(n_layers=4, usegpu=self.usegpu)
        
        return base_cnn
    
    def __generate_enc_stacked_hg(self, input_n_filters, n_levels):
        
        stacked_hourglass = nn.Sequential()
        stacked_hourglass.add_module('Hourglass_1',
                                     RecurrentHourglass(
                                         input_n_filters=input_n_filters,
                                         hidden_n_filters=64,
                                         kernel_size=3,
                                         n_levels=n_levels,
                                         embedding_size=64,
                                         usegpu=self.usegpu))
        stacked_hourglass.add_module('pool_1',
                                     nn.MaxPool2d(2, stride=2))
        stacked_hourglass.add_module('Hourglass_2',
                                     RecurrentHourglass(
                                         input_n_filters=64,
                                         hidden_n_filters=64,
                                         kernel_size=3,
                                         n_levels=n_levels,
                                         embedding_size=64,
                                         usegpu=self.usegpu))        
        stacked_hourglass.add_module('pool_2',
                                     nn.MaxPool2d(2, stride=2))    

        return stacked_hourglass

    def __generate_stacked_renet(self, input_n_filters, n_renets):

        assert n_renets >= 1
        
        renet = nn.Sequential()
        renet.add_module('ReNet_1', ReNet(input_n_filters, 32,
                                          patch_size=(1, 1),
                                          usegpu=self.usegpu))
        for i in range(1, n_renets):
            renet.add_module('ReNet_{}'.format(i + 1),
                             ReNet(32 * 2, 32, patch_size=(1, 1),
                                   usegpu=self.usegpu))
            
        return renet
    
    def __generate_decoder(self, input_n_filters):
        
        decoder = nn.Sequential()
        decoder.add_module('ConvTranspose_1',
                           nn.ConvTranspose2d(input_n_filters,
                                              64,
                                              kernel_size=(2, 2),
                                              stride=(2, 2)))
        decoder.add_module('ReLU_1', nn.ReLU())
        decoder.add_module('ConvTranspose_2',
                           nn.ConvTranspose2d(64,
                                              64,
                                              kernel_size=(2, 2),
                                              stride=(2, 2)))
        decoder.add_module('ReLU_2', nn.ReLU())
        decoder.add_module('Final',
                           nn.ConvTranspose2d(64,
                                              1,
                                              kernel_size=(1, 1),
                                              stride=(1, 1)))
        
        return decoder
            
    def forward(self, x):
        
        x = self.base_cnn(x)
        x = self.enc_stacked_hourglass(x)
        x = self.stacked_renet(x)
        x = self.decoder(x)
        
        return x
