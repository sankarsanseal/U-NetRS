import numpy as np

import os
import sys
import torch
import math

from collections import OrderedDict


from custom_loss import DiceBCELoss


from torch import nn

from torch.nn import functional as F
from torch.autograd import Variable as V

class UnetRS_with_levels(nn.Module):
    """
    
    Dynamically Creating UnetSE with number of given levels

    """

    def __init__(self, in_channels, out_channels, no_of_levels, no_of_outputs = 1, is_bn_enabled = False,\
        dimensions = 3, se_ratio = 1/8):

        """

        U-NetRS initialization parameters

        no_of_levels:   Number of down-sampling ,
                        if we have no_of_levels = 1, we have one level of down-sampling.

                        If we need 4 levels as generic Unet, please set as no_of_levels = 4

        no_of_outputs:  Number of decoder outputs, default is 1.

        is_bn_enable:   BatchNormalization at the each down-sampling levels, default value is False

        dimensions:     Whether modality is 2D or 3D, for 2D please use 2 and for 3D use 3.

        se_ration:      It is used for Sequeeze-and-Excitation module. the "r" value of the ratio.


        """

        super().__init__()
        
        self.starting_conv_channels = 64
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.is_bn_enabled = is_bn_enabled

        self.dimensions = dimensions
        self.se_ratio = se_ratio

        self.no_of_levels = no_of_levels 

        self.no_of_outputs = no_of_outputs

        assert  self.no_of_levels  >=  self.no_of_outputs

        assert self.dimensions in [ 2, 3 ]

        """
        
        
        """


        ###################################################################################

        if self.dimensions == 2:
            self.input_block = nn.BatchNorm2d(num_features = self.in_channels)

        elif self.dimensions == 3:
            self.input_block = nn.BatchNorm3d(num_features = self.in_channels)

        ###########################################################################################
        #
        #  Down Sampling Definition
        #
        ###########################################################################################
        

        self.downward_sampling = nn.ModuleList(
            [ downward_leg ( in_channels=
                                            # No of input channels are selected.
                                            self.in_channels if level == 0\
                                            else  self.starting_conv_channels * 2 ** (level - 1),\
                                            out_channels=\
                                            self.starting_conv_channels * 2 ** (level),\
                                            is_bn_enabled = self.is_bn_enabled,
                                            dimensions = self.dimensions 
                                         ) 

            for level in range(self.no_of_levels)
        ] )




        ###########################################################################################
        #
        # Squeeze-and Excitation Definition
        #
        ###########################################################################################



        self.SE_blocks = nn.ModuleList( [
            Squeeze_and_Excitation(in_channels = self.starting_conv_channels * 2 ** (level),\
                ratio = self.se_ratio, dimensions = self.dimensions)

            for level in range(self.no_of_levels)
        ] )




        ###########################################################################################
        #
        # Attention Gate Definition
        #
        ###########################################################################################


        self.AG_blocks = nn.ModuleList([
            nn.ModuleList([

            Attention_Gate(in_channels = self.starting_conv_channels * 2 ** (level),\
                out_channels = self.starting_conv_channels * 2 ** (level + 1),\
                    dimensions = self.dimensions)

            for level in range(self.no_of_levels - output_index)

            ]) for output_index in reversed(range(self.no_of_outputs))
        ])


        self.bottleneck_backbone = backbone (in_channels = self.starting_conv_channels * \
                                                         2 ** (self.no_of_levels-1),
                                        out_channels = self.starting_conv_channels * \
                                                         2 ** (self.no_of_levels),\
                                        dimensions = self.dimensions)

        self.upward_sampling = nn.ModuleList( [ 
            
            nn.ModuleList([

            upward_leg ( in_channels = self.starting_conv_channels * 2 **(level + 1) ,\
                                        out_channels = self.starting_conv_channels * 2 **(level) ,\
                                        dimensions = self.dimensions

                                        )

                                        for level in range( self.no_of_levels - output_index)

            ]) for output_index in reversed(range(self.no_of_outputs))

        ])

        # self.head = double_conv( in_channels = self.starting_conv_channels * 2,
        #                         out_channels = self.starting_conv_channels)

        if self.dimensions == 2:
            conv_nd = nn.Conv2d

        elif self.dimensions == 3:
            conv_nd = nn.Conv3d
        
        self.mask_combine = nn.ModuleList([
        
            conv_nd( in_channels = self.starting_conv_channels ,\
             out_channels = out_channels ,\
            kernel_size=1, stride=1)

            for output_index in range(self.no_of_outputs)

        ])
        
        self.mask_binary =  nn.ModuleList ( [

            nn.Sigmoid() for output_index in range(self.no_of_outputs)
            
            ])

        # self.mask_selector = Mask_Selector(no_of_outputs = self.no_of_outputs)

        self.dice_score = DiceBCELoss()

        self._init_weights()

        # print(self.downward_sampling, self.bottleneck_backbone,
        #  self.upward_sampling, self.head
        #  )
    
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        for name, layer in self.named_modules():
            if name.endswith("one_to_one_w"):
                nn.init.zeros_(layer.weight.data)
            # if m[0].endswith("one_to_one_w"):
            #     nn.init.zeros_(m)

    def forward(self, input_tensor, ground_truth = None):

        if ground_truth != None:
            
            assert (input_tensor.size(-2), input_tensor.size(-1)) ==\
                 ( ground_truth.size(-2), ground_truth.size(-1))

        down_blocks_list = []

        up_blocks_values_list = [ 
                            [ 
                                [] for level_i in range(self.no_of_levels - out_i) 
                                
                            ] for out_i in reversed(range(self.no_of_outputs))
                                
                        ]

        SE_blocks_list = []

        AG_blocks_values_list = [ 
                            [ 
                                [] for level_i in range(self.no_of_levels - out_i) 
                                
                            ] for out_i in reversed(range(self.no_of_outputs))
                                
                        ]



##########################################################################################################
        """
        
        Down-Sampling Calculation
        
        """
##########################################################################################################

        
        for level_i, down_block in enumerate(self.downward_sampling):

            if level_i == 0 :
                down_blocks_list.append( down_block( input_tensor ) )
                # print(down_blocks_list[-1][0].shape)

            else:
                down_blocks_list.append( down_block ( down_blocks_list[-1][1] ) )

            # print(down_blocks_list[-1][0].shape)

            SE_blocks_list.append( self.SE_blocks [level_i] ( down_blocks_list [level_i] [0] ) )
            # print(SE_blocks_list[-1])


        # print(down_blocks_list[3][0].shape)
        bottleneck_backbone_block = self.bottleneck_backbone(down_blocks_list[-1][1])

        # up_blocks_list.insert(0, bottleneck_backbone_block)


        """

        AG calculation and Up-Sampling

        """

        for output_indx in reversed(range(self.no_of_outputs)):

            for level_i , up_block in enumerate(\
                reversed(self.upward_sampling[output_indx]) 
                ):
                
                
                #[ len ( self.upward_sampling[output_indx] ) : : -1 ]):

                if level_i == 0 and ( output_indx == self.no_of_outputs - 1 ):


                    AG_blocks_values_list\
                        [output_indx]\
                            [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ]\
                                =\
                    self.AG_blocks[output_indx] [ len( self.AG_blocks[output_indx] ) - level_i - 1 ](\
                            input_tensor =\
                                 SE_blocks_list[  len( AG_blocks_values_list[output_indx] ) - level_i -1 ],\
                            low_res_tensor = bottleneck_backbone_block
                                

                           ) 


                #     # AG_block_list[0].insert (0, \
                #     #     self.AG_blocks[output_indx] [ len( self.AG_blocks[output_indx] ) - level_i - 1](\
                #     #         input_tensor = SE_blocks_list[ self.no_of_levels - level_i - 1 ],\
                #     #         low_res_tensor = bottleneck_backbone_block
                #     #             )
                #     # )


                    up_blocks_values_list\
                        [output_indx]\
                            [ len( up_blocks_values_list[output_indx] ) - level_i - 1]\
                    =\
                        up_block ( input_tensor = bottleneck_backbone_block,\
                                concate_with_down = AG_blocks_values_list[output_indx]\
                                    [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ] [0]
                         )

                elif level_i == 0 and  ( output_indx != self.no_of_outputs - 1 ):

                    AG_blocks_values_list[output_indx]\
                        [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ]\
                    =\
                    self.AG_blocks[output_indx] [ len( self.AG_blocks[output_indx] ) - level_i - 1 ](\

                            input_tensor =\
                                 SE_blocks_list[  len( AG_blocks_values_list[output_indx] ) - level_i - 1 ],\
                            low_res_tensor =\
                             down_blocks_list [ len( AG_blocks_values_list[output_indx] ) - level_i] [0]
                                

                           ) 

                    
                    up_blocks_values_list[output_indx]\
                        [ len( up_blocks_values_list[output_indx] ) - level_i - 1]\
                        =\
                            up_block( input_tensor =\
                           down_blocks_list [ len( AG_blocks_values_list[output_indx] ) - level_i] [0],\
                                concate_with_down = AG_blocks_values_list[output_indx] \

                                    [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ] [0]
                         )

                   
                else:


                    AG_blocks_values_list\
                        [output_indx]\
                    [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ]\
                    =\
                        self.AG_blocks[output_indx]\
                                 [ len( self.AG_blocks[output_indx] ) - level_i - 1 ](\

                            input_tensor =\
                            SE_blocks_list[  len( AG_blocks_values_list[output_indx] ) - level_i -1 ],\

                            low_res_tensor =\
                            up_blocks_values_list[output_indx]\
                                [ len( up_blocks_values_list[output_indx] ) - level_i  ]
                                

                           ) 



                    up_blocks_values_list[output_indx]\
                        [ len( up_blocks_values_list[output_indx] ) - level_i - 1]\
                            =\
                    up_block( input_tensor =\
                          up_blocks_values_list[output_indx]\
                              [ len( up_blocks_values_list[output_indx] ) - level_i  ],\
                                concate_with_down = AG_blocks_values_list[output_indx]\
                                    [ len( AG_blocks_values_list[output_indx] ) - level_i - 1 ] [0]
                         )





        # head_out = self.head( torch.cat( [ SE_blocks_list[0],  up_blocks_list[0] ] , dim = 1 ) )

        mask_combined = [ self.mask_combine[output_index] ( up_blocks_values_list[output_index]  [0] ) \

            for output_index in range (self.no_of_outputs)
        
        
         ]

        mask_binary = [ self.mask_binary [ output_index ]( mask_combined [output_index]
        
         ) for output_index in range(self.no_of_outputs) ]


        return mask_binary



        

        



class Unet(nn.Module):

    

    def __init__(self, in_channels:int, out_channels:int, dimensions:int = 2):

        super().__init__()

        self.starting_conv_channels = 64 
        self.dimensions = dimensions

        self.backbone_channels = 1024

        if self.dimensions == 2:

            self.input_block = nn.BatchNorm2d(num_features=in_channels)
            self.conv_nd = nn.Conv2d

        elif self.dimensions == 3:

            self.input_block = nn.BatchNorm3d(num_features=in_channels)
            self.conv_nd = nn.Conv3d

        #############################################################################

        self.down_block1 = downward_leg(in_channels, self.starting_conv_channels,\
            is_bn_enabled= False, dimensions = self.dimensions)

        self.down_block2 = downward_leg(in_channels=self.starting_conv_channels,\
            out_channels = self.starting_conv_channels*2, is_bn_enabled= False, dimensions = self.dimensions )

        self.down_block3 = downward_leg(in_channels =self.starting_conv_channels * 2 ,\
            out_channels = self.starting_conv_channels * 4, is_bn_enabled= False, dimensions = self.dimensions)

        self.down_block4 = downward_leg( in_channels= self.starting_conv_channels * 4  ,\
            out_channels= self.starting_conv_channels * 8, is_bn_enabled= False, dimensions = self.dimensions )

        ##############################################################################


        self.backbone_block = backbone(in_channels=self.starting_conv_channels * 8,\
            out_channels= self.starting_conv_channels * 16, dimensions= self.dimensions )


        ##############################################################################

        self.up_block4 = upward_leg (in_channels= self.starting_conv_channels * 16, \
            out_channels= self.starting_conv_channels * 8, dimensions = self.dimensions)

        self.up_block3 = upward_leg(in_channels=self.starting_conv_channels * 8, \
            out_channels= self.starting_conv_channels * 4, dimensions = self.dimensions)

        self.up_block2 = upward_leg(in_channels=self.starting_conv_channels * 4,\
            out_channels= self.starting_conv_channels * 2, dimensions = self.dimensions)

        self.up_block1 = upward_leg(in_channels=self.starting_conv_channels * 2,\
            out_channels= self.starting_conv_channels, dimensions = self.dimensions)

        ##############################################################################
        


        

        self.mask_combine = self.conv_nd(in_channels= self.starting_conv_channels , out_channels= out_channels,\
            kernel_size=1, stride=1)
        
        self.mask_binary = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_tensor):

        input_bn = self.input_block(input_tensor)

        #######################################################################

        block_conv1, block_down1 = self.down_block1(input_bn)

        block_conv2, block_down2 = self.down_block2(block_down1)

        block_conv3, block_down3 = self.down_block3(block_down2)

        block_conv4, block_down4 = self.down_block4(block_down3)



        block_backbone = self.backbone_block(block_down4)


        #######################################################################


        block_up4 = self.up_block4(block_backbone, block_conv4)



        block_up3 = self.up_block3(block_up4, block_conv3)


        block_up2 = self.up_block2(block_up3, block_conv2)


        block_up1 = self.up_block1(block_up2, block_conv1)



        #######################################################################



        mask_combined = self.mask_combine(block_up1)

        mask_binary = self.mask_binary(mask_combined)

        return mask_binary



class Unet_with_two_levels(nn.Module):

    

    def __init__(self, in_channels:int, out_channels:int, dimensions:int = 2):

        super().__init__()

        self.starting_conv_channels = 64 
        self.dimensions = dimensions

        self.backbone_channels = 1024

        if self.dimensions == 2:

            self.input_block = nn.BatchNorm2d(num_features=in_channels)
            self.conv_nd = nn.Conv2d

        elif self.dimensions == 3:

            self.input_block = nn.BatchNorm3d(num_features=in_channels)
            self.conv_nd = nn.Conv3d

        #############################################################################

        self.down_block1 = downward_leg(in_channels, self.starting_conv_channels,\
            is_bn_enabled= False, dimensions = self.dimensions)

        self.down_block2 = downward_leg(in_channels=self.starting_conv_channels,\
            out_channels = self.starting_conv_channels*2, is_bn_enabled= False, dimensions = self.dimensions )

        ##############################################################################


        self.backbone_block = backbone(in_channels=self.starting_conv_channels * 2,\
            out_channels= self.starting_conv_channels * 4, dimensions= self.dimensions )


        ##############################################################################



        self.up_block2 = upward_leg(in_channels=self.starting_conv_channels * 4,\
            out_channels= self.starting_conv_channels * 2, dimensions = self.dimensions)

        self.up_block1 = upward_leg(in_channels=self.starting_conv_channels * 2,\
            out_channels= self.starting_conv_channels, dimensions = self.dimensions)

        ##############################################################################
        


        

        self.mask_combine = self.conv_nd(in_channels= self.starting_conv_channels , out_channels= out_channels,\
            kernel_size=1, stride=1)
        
        self.mask_binary = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_tensor):

        input_bn = self.input_block(input_tensor)

        #######################################################################

        block_conv1, block_down1 = self.down_block1(input_bn)

        block_conv2, block_down2 = self.down_block2(block_down1)




        block_backbone = self.backbone_block(block_down2)


        #######################################################################




        block_up2 = self.up_block2(block_backbone, block_conv2)



        block_up1 = self.up_block1(block_up2, block_conv1)


        #######################################################################



        mask_combined = self.mask_combine(block_up1)

        mask_binary = self.mask_binary(mask_combined)

        return mask_binary



class UnetSE(nn.Module):

    

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.starting_conv_channels = 64 

        self.backbone_channels = 1024

        self.input_block = nn.BatchNorm2d(num_features=in_channels)

        #############################################################################

        self.down_block1 = downward_leg(in_channels, self.starting_conv_channels)

        self.down_block2 = downward_leg(in_channels=self.starting_conv_channels,\
            out_channels = self.starting_conv_channels*2 )

        self.down_block3 = downward_leg(in_channels =self.starting_conv_channels *2 ,\
            out_channels = self.starting_conv_channels * 4)

        self.down_block4 = downward_leg( in_channels= self.starting_conv_channels * 4  ,\
            out_channels= self.starting_conv_channels * 8 )

        ##############################################################################
        #
        #
        # SE Section
        #
        ##############################################################################


        self.se_block1 = Squeeze_and_Excitation( in_channels= self.starting_conv_channels,\
            ratio = 2)

        self.se_block2 = Squeeze_and_Excitation( in_channels= self.starting_conv_channels * 2,\
            ratio= 2)

        self.se_block3 = Squeeze_and_Excitation( in_channels= self.starting_conv_channels * 4,\
        ratio=2)

        self.se_block4 = Squeeze_and_Excitation( in_channels= self.starting_conv_channels * 8,\
            ratio=2)



        ##############################################################################


        self.backbone_block = backbone(in_channels=self.starting_conv_channels * 8,\
            out_channels= self.starting_conv_channels * 16 )


        ##############################################################################

        self.up_block4 = upward_leg (in_channels= self.starting_conv_channels * 16, \
            out_channels= self.starting_conv_channels * 8)

        self.up_block3 = upward_leg(in_channels=self.starting_conv_channels * 8, \
            out_channels= self.starting_conv_channels * 4)

        self.up_block2 = upward_leg(in_channels=self.starting_conv_channels * 4,\
            out_channels= self.starting_conv_channels * 2)

        self.up_block1 = upward_leg(in_channels=self.starting_conv_channels * 2,\
            out_channels= self.starting_conv_channels)

        ##############################################################################
        
        # self.head_conv2 = nn.Conv2d(in_channels= self.starting_conv_channels * 2,\
        #     out_channels= self.starting_conv_channels, kernel_size=3, stride=1, padding=1,\
        #         padding_mode="replicate")

        # self.head_conv1 = nn.Conv2d(in_channels= self.starting_conv_channels,\
        #      out_channels= self.starting_conv_channels, kernel_size=3 , padding=1,\
        #          padding_mode="replicate")

        

        self.mask_combine = nn.Conv2d(in_channels= self.starting_conv_channels , out_channels= out_channels,\
            kernel_size=1, stride=1)
        
        self.mask_binary = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_tensor):

        input_bn = self.input_block(input_tensor)

        #######################################################################

        block_conv1, block_down1 = self.down_block1(input_bn)

        block_conv2, block_down2 = self.down_block2(block_down1)

        block_conv3, block_down3 = self.down_block3(block_down2)

        block_conv4, block_down4 = self.down_block4(block_down3)

        #######################################################################

        #######################################################################
        #
        #
        # SE block
        #
        #######################################################################

        block_conv1 = self.se_block1(block_conv1)

        block_conv2 = self.se_block2(block_conv2)

        block_conv3 = self.se_block3(block_conv3)

        block_conv4 = self.se_block4(block_conv4)





        #######################################################################


        block_backbone = self.backbone_block(block_down4)



        #######################################################################


        block_up4 = self.up_block4(block_backbone, block_conv4)

        block_up3 = self.up_block3(block_up4, block_conv3)

        block_up2 = self.up_block2(block_up3, block_conv2)

        block_up1 = self.up_block1(block_up2, block_conv1)



        #######################################################################


        # head2 =  self.head_conv2(block_up_concate1)

        # head1 = self.head_conv1(head2)

        mask_combined = self.mask_combine(block_up1)

        mask_binary = self.mask_binary(mask_combined)

        return mask_binary








    
        





class downward_leg(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn_enabled = True, dimensions = 2):

        super().__init__()

        self.is_bn_enabled = is_bn_enabled
        self.dimensions = dimensions



        if self.dimensions == 2:

            self.conv_nd = nn.Conv2d
            self.maxpool_nd = nn.MaxPool2d

            if self.is_bn_enabled:
                self.bn = nn.BatchNorm2d(num_features=in_channels)

        elif self.dimensions == 3:

            self.conv_nd = nn.Conv3d
            self.maxpool_nd = nn.MaxPool3d

            if self.is_bn_enabled:
                self.bn = nn.BatchNorm3d(num_features=in_channels)

        self.conv_section = nn.Sequential(
                OrderedDict([

                ( "conv1", self.conv_nd(in_channels= in_channels, out_channels= out_channels,\
            kernel_size=3, stride=1, padding=1, padding_mode="replicate")),

                ("relu1", nn.ReLU(inplace=True)),
        
                ("conv2", self.conv_nd(in_channels= out_channels, out_channels= out_channels,\
            kernel_size=3, stride=1, padding=1, padding_mode="replicate")),

                ("relu2", nn.ReLU(inplace=True))


                ])

        )
        
        # self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels= out_channels,\
        #     kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        # self.relu1 = nn.ReLU(inplace=True)
        
        # self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels,\
        #     kernel_size=3, stride=1, padding=1, padding_mode="replicate")

        # self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = self.maxpool_nd(2, stride=2)


    def forward(self, input_tensor):

        if self.is_bn_enabled:
            bn_out = self.bn(input_tensor)
        else:
            bn_out = input_tensor


        block_out = self.conv_section(bn_out)


        return block_out , self.maxpool(block_out)



class upward_leg(nn.Module):

    def __init__(self, in_channels, out_channels, dimensions):

        super().__init__()

        self.dimensions = dimensions

        # self.concate_down_channels = concate_down

        if self.dimensions == 2:
            conv_nd = nn.Conv2d

        elif self.dimensions == 3:
            conv_nd = nn.Conv3d
            
            


        
        self.up_conv_high = nn.Sequential(

            OrderedDict([
                ("up_conv_high_up", nn.Upsample(scale_factor=2)),
                ("up_conv_high_conv", conv_nd(in_channels= in_channels,
                 out_channels= in_channels // 2 ,\
            kernel_size=3, stride=1, padding=1, padding_mode="replicate"))


            ])


        )


        self.conv_section = nn.Sequential(

            OrderedDict([

                ("up_conv1", conv_nd(in_channels= in_channels, out_channels= out_channels ,\
            kernel_size=3, stride=1, padding=1, padding_mode="replicate")),

                ("up_relu1", nn.ReLU(inplace=True)),

                ("up_conv2", conv_nd(in_channels= out_channels, out_channels= out_channels,\
            kernel_size=3, stride = 1, padding=1, padding_mode="replicate")),

                ("up_relu2", nn.ReLU(inplace=True))



            ])


        )

        # self.upsample = nn.Upsample(scale_factor=2)


        # self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels= out_channels,\
        #     kernel_size=3, stride=1, padding=1, padding_mode="replicate")

        # self.relu1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,\
        #     kernel_size=3, stride = 1, padding=1, padding_mode="replicate")

        # self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input_tensor, concate_with_down):

        # upsampled_low_res = self.upsample(input_tensor)

        block_out = self.up_conv_high(input_tensor)

        block_out = torch.cat( [concate_with_down, block_out] , dim = 1)

        block_out = self.conv_section(block_out)
        # block_out = self.relu1(block_out)

        # block_out = self.conv2(block_out)
        # block_out = self.relu2(block_out)



        return block_out



class backbone(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, is_bn_enabled:bool = True, dimensions:int = 2):

        super().__init__()

        self.is_bn_enabled = is_bn_enabled
        self.dimensions = dimensions

        if self.dimensions == 2:
            self.bn = nn.BatchNorm2d
            self.conv_nd = nn.Conv2d

        elif self.dimensions == 3:

            self.bn = nn.BatchNorm3d
            self.conv_nd = nn.Conv3d

        self.bb_conv_section = nn.Sequential(
            OrderedDict([
                        ("backbone_bn", self.bn(num_features=in_channels)),
                        ( "backbone_conv1", self.conv_nd(in_channels= in_channels, out_channels=out_channels,
        kernel_size=3, stride=1, padding=1, padding_mode="replicate")),

        ("backbone_conv2",  self.conv_nd(in_channels= out_channels, out_channels=out_channels,
        kernel_size=3, stride=1, padding=1, padding_mode="replicate"))


            ])
        )

        # self.bb_up_conv = nn.Sequential(
        #     OrderedDict([
        #                 ("bb_upsample", nn.UpsamplingNearest2d(scale_factor=2)),
        #                 ("bb_conv_after_up", nn.Conv2d(in_channels= out_channels,\
        #                      out_channels=in_channels,
        # kernel_size=3, stride=1, padding=1, padding_mode="replicate") )

        #     ])
        # )
        # self.bb_upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # self.bb_conv

        # self.backbone_conv1 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels,
        # kernel_size=3, stride=1, padding=1, padding_mode="replicate")

        # self.backbone_conv2 = nn.Conv2d(in_channels= out_channels, out_channels=out_channels,
        # kernel_size=3, stride=1, padding=1, padding_mode="replicate")

    def forward(self, input_tensor):

        block_out = self.bb_conv_section(input_tensor)

        # block_out = self.bb_up_conv(block_out)

        return block_out


class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels, dimensions):

        super().__init__()

        self.dimensions = dimensions

        if self.dimensions == 2:
            self.conv_nd = nn.Conv2d

        elif self.dimensions == 3:
            self.conv_nd = nn.Conv3d

        self.d_conv = nn.Sequential(
                        OrderedDict([

                        ("double_conv1", self.conv_nd(in_channels= in_channels, out_channels=out_channels,
        kernel_size=3, stride=1, padding=1, padding_mode="replicate")),

                        ("double_conv2", self.conv_nd(in_channels= out_channels, out_channels=out_channels,
        kernel_size=3, stride=1, padding=1, padding_mode="replicate"))

                        ])

        )


    def forward(self, input_tensor):

        return self.d_conv(input_tensor)



class Squeeze_and_Excitation(nn.Module):

    def __init__(self, in_channels, ratio, dimensions):

        super().__init__()

        self.out_channels = math.floor(in_channels * ratio)
        self.in_channels = in_channels

        self.dimensions = dimensions

       
        if self.dimensions == 2:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        elif self.dimensions == 3:
            self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        self.selector = nn.Sequential(
            OrderedDict([
                ("se_fc1", nn.Linear(in_features = self.in_channels, out_features= self.out_channels,\
                     bias= False)),
                ("se_relu1", nn.ReLU(inplace=True) ),
                ("se_fc2", nn.Linear(in_features= self.out_channels, out_features = self.in_channels,\
                     bias= False)),
                ("se_sigmoid", nn.Sigmoid())
            ])
        )

        # self.fc1 = nn.Linear(in_features = in_channels, out_features= self.out_channels)

        # self.relu1 = nn.ReLU(inplace=True)

        # self.fc2 = nn.Linear(in_features= self.out_channels, out_features = self.in_channels)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        # global_avg = torch.mean(input_tensor.view(input_tensor.size(0),input_tensor.size(1), -1),\
        #                                         dim=(-1))

        # print(input_tensor.size())

        if self.dimensions == 2:

            b,c, _, _ = input_tensor.size()

        elif self.dimensions == 3:
            
            b,c, _, _, _ = input_tensor.size()


        
        global_avg = self.global_avg_pool(input_tensor)

        #  F.adaptive_avg_pool2d(x, (1, 1))

        # print(input_tensor.shape, global_avg.shape)

        global_avg = global_avg.view(b, c)

        # print("u", global_avg.shape)

        # se_out = self.fc1(global_avg)

        # se_out = self.relu1(se_out)

        # se_out = self.fc2(se_out)

        # se_out = self.sigmoid(se_out)

        se_out = self.selector(global_avg)

        if self.dimensions == 2:

            se_out = se_out.view(b, c, 1, 1 )

        elif self.dimensions == 3:

            se_out = se_out.view(b, c, 1, 1, 1 )


        # print(se_out.shape)

        se_out = se_out.expand_as(input_tensor)

        # print(se_out.shape)

        return input_tensor * se_out


class Attention_Gate(nn.Module):
    """
    This code snippet is taken from Original github of Attention U-net.

    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/grid_attention_layer.py

    """

    def __init__(self, in_channels, out_channels, dimensions):

        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.dimensions = dimensions

        self.sub_sample_kernel_size = 3

        self.sub_sample_factor = 2

        if self.dimensions == 2:

            self.upsample_mode = "bilinear"
            self.conv_nd = nn.Conv2d
            self.bn = nn.BatchNorm2d

        elif self.dimensions == 3:

            self.upsample_mode = "trilinear"
            self.conv_nd = nn.Conv3d
            self.bn = nn.BatchNorm3d

        self.gating_channels = out_channels 
        self.inter_channels = self.gating_channels

        # self.

        self.W = nn.Sequential(
            self.conv_nd(in_channels = self.in_channels, out_channels = self.in_channels,\
                 kernel_size=1, stride=1, padding=0),
            self.bn(self.in_channels),
        )

        self.theta = self.conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor,\
                                  padding=0, bias=False)
        
        self.phi = self.conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = self.conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1,\
             stride=1, padding=0, bias=True)

    def forward(self, input_tensor, low_res_tensor):

        x = input_tensor
        g = low_res_tensor

        # print(x.size(), g.size())

        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode,\
             align_corners= True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, \
           align_corners= True )
        y = sigm_psi_f.expand_as(x) * x

        W_y = self.W(y)

        return W_y , sigm_psi_f

        


class Mask_Selector(nn.Module):

    def __init__(self, no_of_outputs = 1, ratio = 2):

        assert ratio >= 1

        no_of_hidden_nodes = math.floor(no_of_outputs * ratio)

        super().__init__()

        self.no_of_outputs = no_of_outputs
        # self.register_buffer("input_one",  torch.ones(1))
        # self.register_buffer( "input_mlp" , torch.FloatTensor(self.no_of_output)) 

        self.input_one = nn.Parameter(torch.ones(1))
        # self.input_mlp = nn.Parameter(torch.FloatTensor(self.no_of_output))

        
        self.one_to_one = nn.ModuleList(

            [
                nn.Sequential(
                    OrderedDict([
                ( "one_to_one_w", nn.Linear(in_features=1, out_features=1, bias= False)),
                ("one_to_sigmoid" + str( output_index ), nn.Sigmoid() )

                ])
                )


             for output_index in range(no_of_outputs) ]

        )

        # dropout = nn.Dropout(p = 0.2, inplace= True)


        self.mlp = nn.Sequential(
            OrderedDict([
                
                # ("dropout", nn.Dropout(p = 0.5, inplace= True)),


                ("fc1", nn.Linear(in_features = no_of_outputs,\
                     out_features = no_of_hidden_nodes, bias= False)),

                ("relu1", nn.ReLU(inplace=True)),


                ("fc2", nn.Linear( in_features= no_of_hidden_nodes, \
                    out_features = no_of_outputs)),

                ("softmax", nn.Softmax( dim = -1 ))

                

            ])
        )


        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


        for name, layer in self.named_modules():
            if name.endswith("one_to_one_w"):
                nn.init.zeros_(layer.weight.data)

        


        

    def forward(self, input_dice_similarity_score_list):


        assert self.no_of_outputs == len(input_dice_similarity_score_list)

        concatenated_input_to_mlp = None

        one_to_one_values_list = [ [] for output_index in range(self.no_of_outputs) ]

        for output_index in range(self.no_of_outputs) :

         one_to_one_values_list[output_index] = self.one_to_one[output_index](self.input_one) *\
             input_dice_similarity_score_list[output_index]

        # print(self.mlp.fc1.weight.type())

        concatenated_input_to_mlp = torch.cat(\
            [ one_to_one_values_list[output_index]\
                for output_index in range(self.no_of_outputs)
                ], dim  = -1 )


        mlp_output = self.mlp(concatenated_input_to_mlp)

        # print(mlp_output)

        return mlp_output

        



        