import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from glob import glob

# === CONFIGURACION ===
image_dir = r"C:\Users\pabli\Desktop\Juego ascii\imagenes_rectangulares"
save_dir = os.path.join(image_dir, "resultados_generados")
os.makedirs(save_dir, exist_ok=True)

# === HIPERPARÁMETROS ===
image_size = (528, 1328)  # Alto x Ancho
batch_size = 1
epochs = 200
latent_dim = 128
learning_rate = 1e-4

# === TRANSFORMACIONES ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),  # Aumento de datos: voltea horizontalmente con 50% de probabilidad
    transforms.ToTensor(),
])

# === DATASET PERSONALIZADO ===
class MapDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = glob(os.path.join(folder_path, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# === MODELO VAE ===
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 66 * 166, latent_dim)
        self.fc_logvar = nn.Linear(128 * 66 * 166, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 66 * 166)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.view(x.size(0), -1)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        dec_input = self.fc_decode(z).view(x.size(0), 128, 66, 166)
        recon = self.decoder(dec_input)
        return recon, mu, logvar

# === FUNCIONES AUXILIARES ===
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# === ENTRENAMIENTO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MapDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset):.2f}")

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample = torch.randn(4, latent_dim).to(device)
            out = model.decoder(model.fc_decode(sample).view(4, 128, 66, 166))
            save_image(out, os.path.join(save_dir, f"sample_epoch_{epoch+1}.png"))

# === GUARDAR EL MODELO ===
torch.save(model.state_dict(), os.path.join(save_dir, "vae_model.pth"))
