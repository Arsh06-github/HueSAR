import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import warnings

# Suppress rasterio NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for SAR, RGB, and NDVI images
class ImageColorizationDataset(Dataset):
    def __init__(self, dataset_dir, target_size=(256, 256)):
        self.dataset_dir = dataset_dir
        self.target_size = target_size
        self.rgb_dir = os.path.join(dataset_dir, "RGB")
        self.sar_dir = os.path.join(dataset_dir, "SAR")
        self.ndvi_dir = os.path.join(dataset_dir, "NDVI")
        self.image_files = [f for f in os.listdir(self.rgb_dir) if f.endswith('.tif')]
        
        # Filter images with matching SAR and NDVI
        self.valid_files = []
        for img_name in self.image_files:
            rgb_path = os.path.join(self.rgb_dir, img_name)
            sar_path = os.path.join(self.sar_dir, img_name)
            ndvi_path = os.path.join(self.ndvi_dir, img_name)
            if os.path.exists(sar_path) and os.path.exists(ndvi_path):
                self.valid_files.append(img_name)
            else:
                print(f"Skipping {img_name}: Missing SAR or NDVI file")
        self.image_files = self.valid_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load and resize RGB image
        rgb_path = os.path.join(self.rgb_dir, img_name)
        try:
            with rasterio.open(rgb_path) as src:
                rgb = src.read([1, 2, 3]).transpose(1, 2, 0)  # HWC format
                if rgb.shape[2] != 3:
                    raise ValueError(f"RGB image {img_name} has {rgb.shape[2]} channels, expected 3")
                rgb = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
                rgb = rgb.resize(self.target_size, Image.Resampling.LANCZOS)
                rgb = np.array(rgb, dtype=np.float32)
        except Exception as e:
            print(f"Error loading RGB image {img_name}: {e}")
            return None
        
        # Load and resize SAR image
        sar_path = os.path.join(self.sar_dir, img_name)
        try:
            with rasterio.open(sar_path) as src:
                sar = src.read(1)  # Single channel
                sar = Image.fromarray(sar, mode='L')
                sar = sar.resize(self.target_size, Image.Resampling.LANCZOS)
                sar = np.array(sar, dtype=np.float32)
        except Exception as e:
            print(f"Error loading SAR image {img_name}: {e}")
            return None
        
        # Load and resize NDVI image
        ndvi_path = os.path.join(self.ndvi_dir, img_name)
        try:
            with rasterio.open(ndvi_path) as src:
                ndvi = src.read(1)  # Single channel
                ndvi = Image.fromarray(ndvi, mode='L')
                ndvi = ndvi.resize(self.target_size, Image.Resampling.LANCZOS)
                ndvi = np.array(ndvi, dtype=np.float32)
        except Exception as e:
            print(f"Error loading NDVI image {img_name}: {e}")
            return None
        
        # Convert RGB to LAB color space
        lab = rgb2lab(rgb / 255.0)
        l = lab[:, :, 0] / 100.0  # L channel normalized to [0,1]
        ab = lab[:, :, 1:] / 128.0  # AB channels normalized to [-1,1]
        
        # Advanced normalization for SAR
        sar = np.log1p(sar.clip(min=0))  # Log transform
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-8)
        
        # Normalize NDVI
        ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8)
        
        # Stack L channel, SAR, and NDVI as input
        input_img = np.stack([l, sar, ndvi], axis=2)  # Shape: HxWx3
        
        # Convert to torch tensors
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1)).float()  # CHW
        ab = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # CHW (AB channels)
        
        return input_img, ab

# Enhanced U-Net Model for Image Colorization
class UNetColorization(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNetColorization, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Output with residual connection
        out = self.out_conv(d1)
        residual = self.residual_conv(x)
        out = out + 0.1 * residual
        
        return torch.tanh(out)

# Training Function
def train_model(model, dataloader, num_epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        valid_samples = 0
        for inputs, targets in dataloader:
            if inputs is None or targets is None:
                continue  # Skip invalid samples
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            valid_samples += inputs.size(0)
        
        if valid_samples > 0:
            epoch_loss = running_loss / valid_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No valid samples")
    
    return model

# Tkinter GUI for Image Colorization
class ColorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Image Colorization App")
        self.root.geometry("800x600")
        
        # Initialize model
        self.model = UNetColorization(in_channels=3, out_channels=2)
        self.model.to(device)
        
        # Check for pre-trained model
        self.model_path = "colorization_model.pth"
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=device)
                # Check if state_dict matches model architecture
                model_state = self.model.state_dict()
                if all(k in model_state for k in state_dict) and all(
                    model_state[k].shape == state_dict[k].shape for k in state_dict
                ):
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded pre-trained model from {self.model_path}")
                else:
                    print(f"Model architecture mismatch at {self.model_path}. Training new model.")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}. Using new model.")
        else:
            print("No pre-trained model found. Will train a new model when needed.")
        
        self.model.eval()
        
        # GUI elements
        self.label = tk.Label(root, text="Upload a SAR TIF Image to Colorize")
        self.label.pack(pady=10)
        
        self.upload_btn = tk.Button(root, text="Upload SAR Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        
        self.process_btn = tk.Button(root, text="Colorize Image", command=self.colorize_image, state=tk.DISABLED)
        self.process_btn.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=700, height=400)
        self.canvas.pack(pady=10)
        
        self.input_img = None
        self.input_tensor = None
        self.output_img = None
        self.model_input_size = (256, 256)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIF files", "*.tif")])
        if file_path:
            try:
                # Load SAR image
                with rasterio.open(file_path) as src:
                    sar = src.read(1)
                    if len(sar.shape) != 2:
                        raise ValueError(f"SAR image must be single-channel, got shape {sar.shape}")
                    original_size = sar.shape
                    sar = Image.fromarray(sar, mode='L')
                    sar = sar.resize(self.model_input_size, Image.Resampling.LANCZOS)
                    sar = np.array(sar, dtype=np.float32)
                
                # Preprocess SAR
                sar = np.log1p(sar.clip(min=0))
                sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-8)
                
                # Create L channel (approximated from SAR)
                l = sar * 100.0
                ndvi = sar  # Placeholder for NDVI
                
                # Stack channels
                input_img = np.stack([l, sar, ndvi], axis=2)
                
                # Convert to tensor
                input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                self.input_tensor = input_tensor
                self.original_size = original_size
                
                # Display input SAR image
                self.input_img = Image.fromarray((sar * 255).astype(np.uint8), mode='L')
                self.input_img = self.input_img.resize((350, 242), Image.Resampling.LANCZOS)
                self.input_photo = ImageTk.PhotoImage(self.input_img)
                
                self.canvas.delete("all")
                self.canvas.create_image(50, 200, image=self.input_photo, anchor="w")
                self.canvas.create_text(200, 20, text="Input SAR Image", font=("Arial", 12))
                
                self.process_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def colorize_image(self):
        try:
            with torch.no_grad():
                ab = self.model(self.input_tensor)
                ab = ab.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # HxWx2
                l = self.input_tensor.squeeze(0)[0].cpu().numpy()  # L channel
                
                # Resize to original size
                l = Image.fromarray(l * 100.0).resize(self.original_size, Image.Resampling.LANCZOS)
                ab = Image.fromarray(ab * 128.0).resize(self.original_size, Image.Resampling.LANCZOS)
                
                l = np.array(l) / 100.0
                ab = np.array(ab) / 128.0
                
                # Combine L and AB channels
                lab = np.concatenate([l[:, :, np.newaxis], ab], axis=2)
                rgb = lab2rgb(lab) * 255.0
                
                # Convert to uint8
                output = rgb.astype(np.uint8)
                
                # Display output
                self.output_img = Image.fromarray(output, mode='RGB')
                self.output_img = self.output_img.resize((350, 242), Image.Resampling.LANCZOS)
                self.output_photo = ImageTk.PhotoImage(self.output_img)
                
                self.canvas.create_image(400, 200, image=self.output_photo, anchor="w")
                self.canvas.create_text(550, 20, text="Colorized Image", font=("Arial", 12))
                
                # Save output
                self.output_img.save("colorized_output.png")
                messagebox.showinfo("Success", "Image colorized and saved as colorized_output.png")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to colorize image: {e}")

# Main execution
if __name__ == "__main__":
    # Dataset and model setup
    dataset_dir = "C:/Projects/Dataset of Sentinel-1 SAR and Sentinel-2 NDVI Imagery"  # Replace with actual dataset path
    dataset = ImageColorizationDataset(dataset_dir, target_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows
    
    # Check for existing model
    model_path = "colorization_model.pth"
    model = UNetColorization(in_channels=3, out_channels=2)
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model_state = model.state_dict()
            if all(k in model_state for k in state_dict) and all(
                model_state[k].shape == state_dict[k].shape for k in state_dict
            ):
                model.load_state_dict(state_dict)
                print(f"Loaded existing model from {model_path}")
            else:
                print(f"Model architecture mismatch at {model_path}. Training new model.")
                model = train_model(model, dataloader, num_epochs=5)
                torch.save(model.state_dict(), model_path)
                print(f"New model trained and saved as {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Training a new model.")
            model = train_model(model, dataloader, num_epochs=5)
            torch.save(model.state_dict(), model_path)
            print(f"New model trained and saved as {model_path}")
    else:
        print("No existing model found. Training a new model.")
        model = train_model(model, dataloader, num_epochs=5)
        torch.save(model.state_dict(), model_path)
        print(f"New model trained and saved as {model_path}")
    
    # Launch Tkinter GUI with the model
    root = tk.Tk()
    app = ColorizationApp(root)
    app.model.load_state_dict(torch.load(model_path, map_location=device))
    app.model.eval()
    root.mainloop()