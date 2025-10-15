import torch
from torch import nn

# Params to use later
IMG_SIZE = 224
PATCH_SIZE = 16
EMBEDDING_DIM = 768
NUMBER_OF_HEADS = 12
MLP_SIZE = 3072

class ViTPatchEmbedding(nn.Module):
  def __init__(self,
               in_channels:int=2,
               patch_size:int=PATCH_SIZE,
               embedding_dimension:int=EMBEDDING_DIM):
    super().__init__()

    self.patch_size = patch_size

    self.patcher = nn.Conv2d(in_channels=in_channels,
                               out_channels=embedding_dimension,
                               kernel_size=patch_size,
                               stride=patch_size, # Add stride to match kernel_size
                               padding=0)
    self.flatten = nn.Flatten(start_dim=2, end_dim=3)

  def forward(self, x):
    img_resolution = x.shape[-1]
    assert img_resolution % PATCH_SIZE == 0, f"image resolution ({img_resolution}) must be divisible by patch size ({PATCH_SIZE})"
    x_patched = self.patcher(x)
    x_flattened = self.flatten(x_patched)
    return x_flattened.permute(0, 2, 1)

class ViT_arch_class(nn.Module):
  def __init__(self,
               img_size:int=IMG_SIZE,
               patch_size: int=PATCH_SIZE,
               number_transform_layers:int=NUMBER_OF_HEADS,
               embedding_dim:int=EMBEDDING_DIM,
               mlp_size:int=MLP_SIZE,
               number_heads:int=NUMBER_OF_HEADS,
               num_channels:int = 3,
               dropout:float=0.1,
               num_classes:int=1000):
    super().__init__()

    assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}"

    # Correct the number of patches calculation
    num_patch = (img_size // patch_size) * (img_size // patch_size)

    # 1 patch embedding
    self.patch_embedding = ViTPatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dimension=embedding_dim)
    #2 class token
    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

    #3. positional embedding
    self.position_embedding = nn.Parameter(data=torch.randn(1, num_patch+1, embedding_dim), requires_grad=True)

    #4. patch + position embedding dropout
    self.emb_dropout = nn.Dropout(p=dropout)

    #5. Transformer Encoder Layers
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(
        d_model = EMBEDDING_DIM,
        nhead = NUMBER_OF_HEADS,
        dim_feedforward = MLP_SIZE,
        activation = "gelu",
        batch_first = True,
        norm_first = True),
        num_layers = number_transform_layers)

     #7 MLP head
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim, out_features=num_classes))

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.patch_embedding(x)
    class_token = self.class_token.expand(batch_size, -1, -1)
    x = torch.cat((class_token, x), dim=1)
    x = self.position_embedding + x
    x = self.emb_dropout(x)
    x = self.transformer_encoder(x)
    x = self.mlp_head(x[:, 0])
    return x