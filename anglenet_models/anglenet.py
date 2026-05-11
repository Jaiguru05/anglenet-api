import torch
import torch.nn as nn
import timm

# ============================================================
# CBAM
# ============================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super(CBAM, self).__init__()

        # ====================================================
        # CHANNEL ATTENTION
        # ====================================================

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // 16,
                1,
                bias=False
            ),

            nn.ReLU(),

            nn.Conv2d(
                channels // 16,
                channels,
                1,
                bias=False
            )
        )

        self.sigmoid_channel = nn.Sigmoid()

        # ====================================================
        # SPATIAL ATTENTION
        # ====================================================

        self.spatial = nn.Conv2d(

            2,

            1,

            kernel_size=7,

            padding=3,

            bias=False
        )

        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):

        # ====================================================
        # CHANNEL ATTENTION
        # ====================================================

        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        channel_attention = self.sigmoid_channel(
            avg_out + max_out
        )

        x = x * channel_attention

        # ====================================================
        # SPATIAL ATTENTION
        # ====================================================

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        spatial_attention = self.sigmoid_spatial(

            self.spatial(
                spatial_input
            )
        )

        x = x * spatial_attention

        return x

# ============================================================
# ANGLENET
# ============================================================

class AngleNet(nn.Module):

    def __init__(self):

        super(AngleNet, self).__init__()

        # ====================================================
        # EFFICIENTNET B3
        # ====================================================

        self.backbone = timm.create_model(

            "efficientnet_b3",

            pretrained=True,

            num_classes=0,

            in_chans=4
        )

        feature_dim = self.backbone.num_features

        # ====================================================
        # CBAM
        # ====================================================

        self.cbam = CBAM(
            feature_dim
        )

        # ====================================================
        # SHARED FEATURES
        # ====================================================

        self.shared = nn.Sequential(

            nn.Linear(
                feature_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        # ====================================================
        # REGRESSION HEAD
        # ====================================================

        self.regressor = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                1
            )
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                4
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        features = self.backbone.forward_features(
            x
        )

        features = self.cbam(
            features
        )

        features = nn.functional.adaptive_avg_pool2d(
            features,
            1
        )

        features = features.view(
            features.size(0),
            -1
        )

        shared_features = self.shared(
            features
        )

        angle = self.regressor(
            shared_features
        )

        severity = self.classifier(
            shared_features
        )

        return angle, severity