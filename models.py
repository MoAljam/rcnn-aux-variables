# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: not the cleanest implementation, but follows the gist of the architecture and enables easier analysis

# Category-orthogonal object features guide information processing in recurrent neural networks trained for object categorization
# FF
# input (64, 64, 1)
# conv_1
# (64, 64, 1) -> conv2d(out_c=32, k=7, s=4) -> relu -> shape (16, 16, 32), E_RF =  7 , E_St = 4
# conv_2
# (16, 16, 32) -> conv2d(out_c=64, k=3, s=2) -> relu -> shape (8, 8, 64), E_R = 7 + (3-1)*4 = 15 , E_St = 4*2 = 8
# conv_3
# (8, 8, 64) -> conv2d(out_c=10+10, k=3, s=1) -> relu -> shape (8, 8, 20), E_R = 15 + (3-1)*8 = 31 , E_St = 8*1 = 8
# avgpool on first 10 for output classes
# (8, 8, 20) -> avgpool on first 10 channels -> shape (1, 1, 10)

# # recurrent
# top_down
# conv_2 -> input, trasposeConv2d(in_c=64, out_c=1, k=9, s=8, p=1, output_p=1)
# conv_3 -> input, trasposeConv2d(in_c=20, out_c=1, k=9, s=8, p=1, output_p=1)
# lateral
# conv_2 -> conv_2, conv2d(in_c=64, out_c=64, k=3, s=1,)
# conv_3 -> conv_3, conv2d(in_c=20, out_c=20, k=3, s=1,)


class RCNN(nn.Module):
    def __init__(self, num_classes=10, modulation_type="multiplicative"):
        super().__init__()
        self.num_classes = num_classes
        self.modulation_type = modulation_type

        # FF
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=3)  # (B,32,16,16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (B,64,8,8)
        self.conv3 = nn.Conv2d(64, num_classes + 10, kernel_size=3, stride=1, padding=1)  # (B,10+10,8,8)
        # RNN
        # self.conv1_lateral = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.conv2_lateral = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv3_lateral = nn.Conv2d(num_classes + 10, num_classes + 10, kernel_size=3, padding="same")

        # self.deconv1_to_input = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.deconv2_to_input = nn.ConvTranspose2d(64, 1, kernel_size=9, stride=8, padding=1, output_padding=1)
        self.deconv3_to_input = nn.ConvTranspose2d(
            num_classes + 10, 1, kernel_size=9, stride=8, padding=1, output_padding=1
        )

        # self.deconv2_to_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv3_to_conv2 = nn.ConvTranspose2d(
        #     num_classes + 10, 64, kernel_size=3, stride=1, padding=1, output_padding=0
        # )

    # TODO enable pernurbations analysis
    def forward(self, x, timesteps=5, return_actvs=False, pernurbations_inputs: dict = None):

        activations = {}

        if "input" not in activations:
            activations["input"] = []
            activations["input"].append(x)
        if "conv1" not in activations:
            activations["conv1"] = []
            activations["conv1"].append(F.relu(self.conv1(activations["input"][0])))
        if "conv2" not in activations:
            activations["conv2"] = []
            activations["conv2"].append(F.relu(self.conv2(activations["conv1"][0])))
        if "conv3" not in activations:
            activations["conv3"] = []
            activations["conv3"].append(F.relu(self.conv3(activations["conv2"][0])))
        if "output" not in activations:
            activations["output"] = []
            activations["output"].append(
                F.adaptive_avg_pool2d(activations["conv3"][0][:, : self.num_classes, :, :], (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )
        if "aux_output" not in activations:
            activations["aux_output"] = []
            activations["aux_output"].append(
                F.adaptive_avg_pool2d(activations["conv3"][0][:, self.num_classes :, :, :], (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )

        for t in range(1, timesteps):
            # acts from t-1 for recurrent connections
            conv1_recurrent_T = None
            conv2_recurrent_T = None
            conv3_recurrent_T = None
            output_recurrent_T = None
            aux_output_recurrent_T = None
            # check pernurbations
            if pernurbations_inputs is not None and t - 1 in pernurbations_inputs:
                x = pernurbations_inputs[t - 1].get("input", x)
                conv1_recurrent_T = pernurbations_inputs[t - 1].get("conv1", activations["conv1"][t - 1])
                conv2_recurrent_T = pernurbations_inputs[t - 1].get("conv2", activations["conv2"][t - 1])
                conv3_recurrent_T = pernurbations_inputs[t - 1].get("conv3", activations["conv3"][t - 1])
                output_recurrent_T = pernurbations_inputs[t - 1].get("output", activations["output"][t - 1])
                aux_output_recurrent_T = pernurbations_inputs[t - 1].get("aux_output", activations["aux_output"][t - 1])
            else:
                conv1_recurrent_T = activations["conv1"][t - 1]
                conv2_recurrent_T = activations["conv2"][t - 1]
                conv3_recurrent_T = activations["conv3"][t - 1]
                output_recurrent_T = activations["output"][t - 1]
                aux_output_recurrent_T = activations["aux_output"][t - 1]

            # Reconstruct input from conv3 and conv2
            # recon_from_conv2 = self.deconv2_to_input(activations["conv2"][t - 1])
            # recon_from_conv3 = self.deconv3_to_input(activations["conv3"][t - 1])

            recon_from_conv2 = self.deconv2_to_input(conv2_recurrent_T)
            recon_from_conv3 = self.deconv3_to_input(conv3_recurrent_T)

            recon_input = torch.sigmoid(recon_from_conv3 + recon_from_conv2)

            activations["input"].append(
                F.relu(x + (2 * recon_input - 1) if self.modulation_type == "additive" else x * (2 * recon_input))
            )

            # Forward pass
            activations["conv1"].append(F.relu(self.conv1(activations["input"][t])))
            conv2_ff_out = self.conv2(activations["conv1"][t])
            # conv2_l_out = self.conv2_lateral(activations["conv2"][t - 1])
            conv2_l_out = self.conv2_lateral(conv2_recurrent_T)
            activations["conv2"].append(
                F.relu(conv2_ff_out + conv2_l_out)
                if self.modulation_type == "additive"
                else F.relu(conv2_ff_out) * (2 * torch.sigmoid(conv2_l_out))
            )
            conv3_ff_out = self.conv3(activations["conv2"][t])
            # conv3_l_out = self.conv3_lateral(activations["conv3"][t - 1])
            conv3_l_out = self.conv3_lateral(conv3_recurrent_T)
            activations["conv3"].append(
                F.relu(conv3_ff_out + conv3_l_out)
                if self.modulation_type == "additive"
                else F.relu(conv3_ff_out) * (2 * torch.sigmoid(conv3_l_out))
            )

            # Readout
            activations["output"].append(
                F.adaptive_avg_pool2d(activations["conv3"][t][:, : self.num_classes, :, :], (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )
            activations["aux_output"].append(
                F.adaptive_avg_pool2d(activations["conv3"][t][:, self.num_classes :, :, :], (1, 1))
                .squeeze(-1)
                .squeeze(-1)
            )

        if return_actvs:
            return activations
        else:
            return activations["output"]


# %%
if __name__ == "__main__":
    # test the trasnpose conv layers shapes with dummy data
    model = RCNN(num_classes=10, modulation_type="multiplicative")

    # test deconv1_to_input
    dummy_conv1 = torch.randn(1, 32, 16, 16)
    recon_input_from_conv1 = model.deconv1_to_input(dummy_conv1)
    print("shape output from conv1:", dummy_conv1.shape)
    print(f"Reconstructed input from conv1 shape: {recon_input_from_conv1.shape}")  # expect (1, 1, 64, 64)
    # deconv2_to_input
    dummy_conv2 = torch.randn(1, 64, 8, 8)
    recon_input_from_conv2 = model.deconv2_to_input(dummy_conv2)
    print("shape output from conv2:", dummy_conv2.shape)
    print(f"Reconstructed input from conv2 shape: {recon_input_from_conv2.shape}")  # expect (1, 1, 64, 64)
    # deconv3_to_input
    dummy_conv3 = torch.randn(1, 20, 8, 8)
    recon_input_from_conv3 = model.deconv3_to_input(dummy_conv3)
    print("shape output from conv3:", dummy_conv3.shape)
    print(f"Reconstructed input from conv3 shape: {recon_input_from_conv3.shape}")  # expect (1, 1, 64, 64)
    # deconv2_to_conv1
    recon_conv1_from_conv2 = model.deconv2_to_conv1(dummy_conv2)
    print(f"Reconstructed conv1 from conv2 shape: {recon_conv1_from_conv2.shape}")  # expect (1, 32, 16, 16)
    # deconv3_to_conv2
    recon_conv2_from_conv3 = model.deconv3_to_conv2(dummy_conv3)
    print(f"Reconstructed conv2 from conv3 shape: {recon_conv2_from_conv3.shape}")  # expect (1, 64, 8, 8)

# %%
