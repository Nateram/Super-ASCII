# Nota: Instala la librería requests si no la tienes: pip install requests

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import json
import requests
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Configuración de Gemini API
GEMINI_API_KEY = "AIzaSyCoFRhRa1zamcMIqIxIuBChtehsoRE5AUM"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
QUALITY_THRESHOLD = 50  # Puntuación mínima para detener entrenamiento
CONSECUTIVE_MAPS = 3    # Mapas consecutivos que deben superar el umbral
RATINGS_FILE = "map_ratings.json"  # Archivo para guardar valoraciones

# Constantes para elementos del juego
JUGADOR = 'J'
META = '|>'
SUELO = '■'
AGUA = '🟦'
MONEDA = 'C'
BLOQUE = '□'
ENEMIGO = '☉'

# Constantes para codificación de caracteres
CHAR_MAP = {
    ' ': 0,  # Espacio vacío
    JUGADOR: 1,
    '|': 2,
    '>': 3,
    SUELO: 4,
    AGUA: 5,
    MONEDA: 6,
    BLOQUE: 7,
    ENEMIGO: 8,
    '/': 9,
    '\\': 10,
    '\n': 11,  # Nueva línea
    '<PAD>': 12,  # Padding
}

# Decodificación (índice a carácter)
IDX_MAP = {v: k for k, v in CHAR_MAP.items()}

# Configuración del modelo
MAP_HEIGHT = 16  # Alto máximo del mapa  
MAP_WIDTH = 80   # Ancho máximo del mapa
EMBED_DIM = 128  # Dimensión de embedding para caracteres
BATCH_SIZE = 4   # Tamaño del batch
EPOCHS = 30      # Épocas de entrenamiento (máximo)
LEARNING_RATE = 2e-4  # Tasa de aprendizaje
NUM_MAPS_GEN = 8      # Mapas a generar al final
DIFFUSION_STEPS = 500  # Pasos del proceso de difusión
EVAL_FREQUENCY = 2      # Evaluar cada X épocas

class MapRatingSystem:
    """Sistema para valorar mapas según su similitud con los mapas originales"""
    def __init__(self, ratings_file=RATINGS_FILE):
        self.ratings_file = ratings_file
        self.ratings = self._load_ratings()
        self.gemini_client = self._init_gemini_client()
        self.system_prompt = self._create_system_prompt()
        self.reference_maps = self._load_reference_maps()
        
    def _load_ratings(self):
        """Cargar valoraciones existentes o crear un nuevo archivo"""
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print(f"Error al cargar {self.ratings_file}, creando nuevo archivo")
        return {}
    
    def _load_reference_maps(self):
        """Cargar los 10 primeros mapas originales como referencia"""
        maps = []
        for i in range(1, 11):
            try:
                map_file = f"mapa{i:02d}.txt"  # Formato: mapa01.txt, mapa02.txt, etc.
                if os.path.exists(map_file):
                    with open(map_file, 'r', encoding='utf-8') as f:
                        maps.append(f.read())
            except Exception as e:
                print(f"Error al cargar mapa de referencia {i}: {e}")
        
        if maps:
            print(f"Cargados {len(maps)} mapas de referencia")
        else:
            print("⚠️ No se encontraron mapas de referencia")
            
        return maps
    
    def save_ratings(self):
        """Guardar valoraciones a archivo JSON"""
        with open(self.ratings_file, 'w', encoding='utf-8') as f:
            json.dump(self.ratings, f, indent=2)
    
    def _init_gemini_client(self):
        """Inicializar cliente Gemini API"""
        try:
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
                print("ERROR: API Key de Gemini no configurada correctamente.")
                return None
            return True  # Just a flag to indicate API key is available
        except Exception as e:
            print(f"Error al inicializar Gemini API: {e}")
            return None
    
    def _create_system_prompt(self):
        """Crear prompt para Gemini con ejemplos de mapas perfectos"""
        prompt = """Eres un evaluador experto de mapas de plataformas ASCII para un juego tipo Super Mario Bros.

Te presentaré primero 10 mapas perfectos que sirven como referencia del estilo y estructura correctos.
Luego, cuando te presente un nuevo mapa, deberás evaluarlo en una escala de 1 a 100.
- 100 significa que es idéntico en estilo y calidad a los mapas de referencia
- 1 significa que no se parece en nada a los mapas de referencia

Elementos del mapa:
- J: Jugador
- |>: Meta/Bandera
- ■: Bloques/Plataformas/Suelo
- 🟦: Agua
- C: Monedas
- □: Bloques sorpresa
- ☉: Enemigos

Considera los siguientes factores para tu evaluación:
1. Presencia de elementos esenciales (jugador J y meta |> con | hacia el abajo)
2. Estructura similar a los mapas de referencia
3. Distribución equilibrada de elementos decorativos
4. Posicionamiento lógico de plataformas y elementos
5. Coherencia visual y estética general
6. Para un mapa jugable con minimo J linea recta de bloques y meta al final la valoración será 50 

Proporciona solo una puntuación numérica del 1 al 100, sin explicación adicional.
"""
        return prompt
    
    def _make_gemini_request(self, prompt):
        """Hacer petición a la API de Gemini"""
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 100
            }
        }
        
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                return content.strip()
            else:
                print(f"Respuesta inesperada de Gemini: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error en petición a Gemini: {e}")
            return None
        except Exception as e:
            print(f"Error procesando respuesta de Gemini: {e}")
            return None
    
    def rate_map(self, map_content=None, map_file=None):
        """Evaluar un mapa usando Gemini API comparándolo con los de referencia"""
        # Verificar si hay contenido de mapa
        if map_content is None and map_file is None:
            raise ValueError("Debes proporcionar contenido de mapa o un archivo")
        
        # Si se proporciona un archivo, cargar su contenido
        if map_content is None:
            map_path = os.path.abspath(map_file)
            try:
                with open(map_path, 'r', encoding='utf-8') as f:
                    map_content = f.read()
            except Exception as e:
                print(f"Error leyendo archivo {map_file}: {e}")
                return 50  # Valor por defecto
        
        # Verificar si ya tenemos valoración para este mapa
        map_hash = hash(map_content)
        if str(map_hash) in self.ratings:
            return self.ratings[str(map_hash)]
        
        # Si no hay cliente Gemini o no hay mapas de referencia, usar valoración fallback
        if self.gemini_client is None or not self.reference_maps:
            score = self._fallback_rating(map_content)
            self.ratings[str(map_hash)] = score
            self.save_ratings()
            return score
        
        # Evaluar con Gemini
        try:
            # Crear prompt con mapas de referencia
            full_prompt = self.system_prompt + "\n\n"
            full_prompt += "A continuación te muestro los 10 mapas perfectos de referencia:\n\n"
            
            for i, ref_map in enumerate(self.reference_maps):
                full_prompt += f"MAPA REFERENCIA {i+1}:\n```\n{ref_map}\n```\n\n"
            
            full_prompt += "Ahora, evalúa este nuevo mapa con una puntuación de 1 a 100:\n\n```\n" + map_content + "\n```\n"
            
            response_text = self._make_gemini_request(full_prompt)
            
            if response_text is None:
                # Si la API falla, usar valoración fallback
                score = self._fallback_rating(map_content)
                self.ratings[str(map_hash)] = score
                self.save_ratings()
                return score
            # Intentar extraer un número de la respuesta
            # Buscar un número en la respuesta
            import re
            score_match = re.search(r'\b(\d{1,3})\b', response_text)
            
            if score_match:
                score = int(score_match.group(1))
                # Validar que está en rango 1-100
                score = max(1, min(100, score))
            else:
                # Si no hay número, usar valoración por defecto
                score = self._fallback_rating(map_content)
            
            # Guardar y devolver puntuación
            self.ratings[str(map_hash)] = score
            self.save_ratings()
            return score
            
        except Exception as e:
            print(f"Error en evaluación Gemini: {e}")
            score = self._fallback_rating(map_content)
            self.ratings[str(map_hash)] = score
            self.save_ratings()
            return score
    
    def _fallback_rating(self, map_content):
        """Método alternativo de puntuación si Gemini falla"""
        # Detectar patrones inválidos o extremadamente simples
        lines = map_content.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        
        # Si todos los caracteres son iguales, es un mapa muy malo
        if len(non_empty_lines) > 0 and all(line == non_empty_lines[0] for line in non_empty_lines):
            return 5  # Muy mala calidad
        
        # Base de puntuación
        score = 30  # Puntuación base
        
        # Penalizaciones para elementos críticos faltantes
        if JUGADOR not in map_content:
            score -= 30
        if META not in map_content:
            score -= 30
        if SUELO not in map_content:
            score -= 20
        
        # Verificar estructura mínima
        has_structure = False
        for i in range(len(lines)-1):
            if i < len(lines)-1 and JUGADOR in lines[i] and SUELO in lines[i+1]:
                has_structure = True
                break
        
        if not has_structure:
            score -= 15  # Penalizar falta de estructura básica
        
        # Premiar variedad
        if MONEDA in map_content:
            score += 10
        if ENEMIGO in map_content:
            score += 10
        if BLOQUE in map_content:
            score += 10
        if AGUA in map_content:
            score += 10
        
        # Formato y longitud adecuados
        if len(lines) >= 3 and all(not line.strip() for line in lines[:3]):
            score += 15
        
        if len(lines) > 10:
            score += 15
        
        # Evaluar distribución
        if map_content.count(SUELO) > 5 and map_content.count(SUELO) < 200:
            score += 10  # Premio por cantidad razonable de bloques
        
        # Asegurar que el puntaje esté en el rango [1, 100]
        return max(1, min(100, score))

class MapDataset(Dataset):
    """Dataset para mapas ASCII convertidos a representación 2D para difusión"""
    def __init__(self, map_files, max_height=MAP_HEIGHT, max_width=MAP_WIDTH):
        self.examples = []
        self.map_files = map_files
        self.max_height = max_height
        self.max_width = max_width
        
        template_map = """

J
■■■■  □  C   
      ■■■■    
                                    |>
                                   /■\\
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■"""
        
        # Guardar mapa plantilla para referencia
        self.template = self._text_to_2d_map(template_map)
        
        # Cargar mapas
        for file in map_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Asegurar que el mapa tenga elementos esenciales
                has_player = JUGADOR in content
                has_goal = META in content
                has_floor = SUELO in content
                
                if has_player and has_goal and has_floor:
                    # Convertir texto a grid 2D
                    map_grid = self._text_to_2d_map(content)
                    self.examples.append(map_grid)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Loaded {len(self.examples)} maps for diffusion model")
    
    def _text_to_2d_map(self, map_text):
        """Convierte texto ASCII a matriz 2D de índices de caracteres"""
        lines = map_text.split('\n')
        
        # Asegurar formato consistente
        while len(lines) < 3 or lines[0].strip() or lines[1].strip() or lines[2].strip():
            lines.insert(0, '')
        
        map_grid = torch.zeros((self.max_height, self.max_width), dtype=torch.long)
        
        # Copiar caracteres del mapa al grid
        for i, line in enumerate(lines[:self.max_height]):
            for j, char in enumerate(line[:self.max_width]):
                map_grid[i, j] = CHAR_MAP.get(char, CHAR_MAP[' '])
        
        # Asegurar que hay caracteres esenciales
        player_present = (map_grid == CHAR_MAP[JUGADOR]).any().item()
        goal_present = ((map_grid == CHAR_MAP['|']) & (map_grid.roll(-1, dims=1) == CHAR_MAP['>'])).any().item()
        floor_present = (map_grid == CHAR_MAP[SUELO]).any().item()
        
        if not (player_present and goal_present and floor_present):
            # Si falta algo esencial, usa la plantilla
            print(f"Warning: Map missing essential elements, using template")
            map_grid = self.template.clone()
        
        return map_grid
    
    def _2d_map_to_text(self, map_grid):
        """Convierte grid de índices a texto ASCII"""
        lines = []
        for i in range(map_grid.shape[0]):
            line = ''.join([IDX_MAP.get(idx.item(), ' ') for idx in map_grid[i]])
            lines.append(line.rstrip())
        
        # Eliminar líneas vacías al final
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        map_grid = self.examples[idx]
        # Convertir a float para el modelo de difusión
        map_grid_float = F.one_hot(map_grid, num_classes=len(CHAR_MAP)).float()
        return map_grid_float.permute(2, 0, 1)  # [C, H, W] para convoluciones

class SimpleUNet(nn.Module):
    """Arquitectura U-Net básica para modelo de difusión"""
    def __init__(self, in_channels=len(CHAR_MAP), out_channels=len(CHAR_MAP)):
        super().__init__()
        
        # Dimensión de tiempo embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        
        # Camino de codificación (downsample)
        self.conv1 = self._block(in_channels, 64)
        self.conv2 = self._block(64, 128)
        self.conv3 = self._block(128, 256)
        
        # Capa de cuello de botella
        self.bottleneck = self._block(256, 512)
        
        # Camino de decodificación (upsample)
        self.upconv3 = self._block(512 + 256, 256)
        self.upconv2 = self._block(256 + 128, 128)
        self.upconv1 = self._block(128 + 64, 64)
        
        # Capa final
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling y upsampling
        self.pool = nn.MaxPool2d(2)
        # Ya no usaremos este upsample fijo
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _block(self, in_channels, out_channels):
        """Bloque convolucional básico"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, t):
        """
        Forward pass de U-Net
        x: [B, C, H, W] - Tensor de entrada (mapa con ruido)
        t: [B] - Paso de tiempo en el proceso de difusión
        """
        # Time embedding - Convertir t a float antes de procesarlo
        t = t.float()  # Asegurar que t es de tipo float
        temb = self.time_embedding(t.unsqueeze(-1))  # [B, 128]
        
        # Añadir time embedding como canal adicional
        temb = temb.unsqueeze(-1).unsqueeze(-1)  # [B, 128, 1, 1]
        
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        
        # Bottleneck - añadir time embedding
        x = self.bottleneck(self.pool(x3))
        
        # Decoder con skip connections - usando interpolación con tamaños exactos
        # Upsampling 1: Asegurar que coincida exactamente con x3
        x = F.interpolate(x, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv3(x)
        
        # Upsampling 2: Asegurar que coincida exactamente con x2
        x = F.interpolate(x, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)
        
        # Upsampling 3: Asegurar que coincida exactamente con x1
        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv1(x)
        
        # Capa de salida
        return self.output_conv(x)


class GaussianDiffusion:
    """Implementación de difusión gaussiana para mapas"""
    def __init__(self, num_diffusion_steps=DIFFUSION_STEPS, device='cpu'):
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Definir programa beta para añadir ruido y moverlo al dispositivo correcto
        self.beta = self._cosine_beta_schedule(num_diffusion_steps).to(device)
        self.alpha = (1 - self.beta).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Programa coseno para betas (produce mejores resultados)
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_0, t):
        """
        Función para añadir ruido a x_0 en el paso t
        - x_0: Datos originales [B, C, H, W]
        - t: Pasos de tiempo [B]
        
        Retorna x_t (datos con ruido) y el ruido añadido
        """
        epsilon = torch.randn_like(x_0)  # Ruido gaussiano
        
        # Extraer alpha_bar para estos pasos t - ya está en el dispositivo correcto
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        
        # Añadir ruido (ecuación de difusión)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        
        return x_t, epsilon
    
    def sample(self, model, shape, device, guide_map=None, class_guidance_scale=3.0):
        """
        Genera una muestra usando el modelo y el proceso de denoising
        - guide_map: Mapa guía opcional para generación condicional
        - class_guidance_scale: Factor de guía (>1 para generación más guiada)
        """
        # Empezar con ruido puro
        x = torch.randn(shape, device=device)
        
        # Crear estructura de mapa base para guiar la generación
        if guide_map is None:
            # Crear mapa guía con estructura mínima
            guide_map = self._create_guide_map(shape, device)
        
        # Proceso de denoising paso por paso
        for t in tqdm(reversed(range(self.num_diffusion_steps)), desc='Sampling'):
            t_tensor = torch.tensor([t], device=device).repeat(shape[0])
            
            # Añadir ruido al tensor para evitar determinismo
            noise = torch.randn_like(x) if t > 0 else 0
            
            # Paso básico de predicción
            with torch.no_grad():
                # Predicción estándar
                predicted_noise = model(x, t_tensor)
                
                # Si tenemos mapa guía, realizar guidance
                if guide_map is not None:
                    # Mezclar el mapa guía con x basado en paso t
                    guidance_strength = class_guidance_scale * (1.0 - self.alpha_bar[t].item())
                    x = x * (1 - guidance_strength) + guide_map * guidance_strength
            
            # Calcular estimación de x_0 (todos los tensores ya están en el dispositivo correcto)
            alpha_bar_t = self.alpha_bar[t]
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]
            
            if t > 0:
                # Fórmula para paso t > 0
                noise_coef = beta_t / torch.sqrt(1 - alpha_bar_t)
                x = (x - noise_coef * predicted_noise) / torch.sqrt(alpha_t)
                x = x + torch.sqrt(beta_t) * noise
            else:
                # Último paso (t=0)
                x = (x - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        return x
    

    def _create_guide_map(self, shape, device):
        """Crea un mapa guía básico con estructuras esenciales"""
        # Extraer dimensiones
        B, C, H, W = shape
        guide_map = torch.zeros(shape, device=device)
        
        # Definir posiciones para elementos clave
        # Jugador arriba a la izquierda
        player_h, player_w = H // 4, W // 10
        # Meta arriba a la derecha
        goal_h, goal_w = H // 4, W * 9 // 10
        # Suelo en la parte inferior
        floor_h = 3 * H // 4
        
        # Definir índices de canales para elementos clave
        player_idx = CHAR_MAP[JUGADOR]
        pipe_idx = CHAR_MAP['|']
        gt_idx = CHAR_MAP['>']
        floor_idx = CHAR_MAP[SUELO]
        
        # Colocar elementos en el mapa guía
        # Jugador
        guide_map[:, player_idx, player_h, player_w] = 1.0
        # Meta
        guide_map[:, pipe_idx, goal_h, goal_w] = 1.0
        guide_map[:, gt_idx, goal_h, goal_w + 1] = 1.0
        # Suelo
        for w in range(W):
            guide_map[:, floor_idx, floor_h, w] = 0.8
        
        return guide_map

def train_diffusion_model(model, dataloader, diffusion, optimizer, device, epochs=EPOCHS):
    """Entrena el modelo de difusión"""
    # Historial de pérdida
    loss_history = []
    
    # Bucle de entrenamiento
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # Training
        model.train()
        epoch_losses = []
        train_bar = tqdm(dataloader, desc="Training")
        
        for batch in train_bar:
            # Mover datos al dispositivo
            maps = batch.to(device)
            B = maps.shape[0]
            
            # Seleccionar timesteps aleatorios
            t = torch.randint(0, diffusion.num_diffusion_steps, (B,), device=device)
            
            # Añadir ruido según timestep
            noisy_maps, target_noise = diffusion.add_noise(maps, t)
            
            # Predecir ruido con el modelo
            predicted_noise = model(noisy_maps, t)
            
            # Calcular pérdida
            loss = F.mse_loss(predicted_noise, target_noise)
            
            # Optimizar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Registrar pérdida
            epoch_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Pérdida media para esta época
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        # Guardar modelo cada 5 épocas
        if epoch % 5 == 0 or epoch == epochs:
            model_path = f"diffusion_model_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            print(f"Model saved as {model_path}")
        
        # Generar mapa de muestra para ver progreso
        if epoch % 2 == 0:
            print("\nGenerando mapa de muestra para verificar progreso...")
            sample_map = generate_map_sample(model, diffusion, device, 1)
            print_sample_map(sample_map[0])
    
    return loss_history

def generate_map_sample(model, diffusion, device, batch_size=1, guide_strength=2.0):
    """Genera un mapa de muestra usando el modelo de difusión entrenado"""
    model.eval()
    
    # Dimensiones para la generación
    shape = (batch_size, len(CHAR_MAP), MAP_HEIGHT, MAP_WIDTH)
    
    # Genera muestra usando el proceso de difusión inverso
    samples = diffusion.sample(model, shape, device, class_guidance_scale=guide_strength)
    
    # Convertir a índices de caracteres (one-hot a índices)
    map_indices = torch.argmax(samples, dim=1)
    
    return map_indices

def print_sample_map(map_indices):
    """Imprime una vista previa del mapa generado"""
    map_text = ""
    for i in range(min(10, map_indices.shape[0])):  # Imprime solo las primeras 10 líneas
        line = ''.join([IDX_MAP.get(idx.item(), ' ') for idx in map_indices[i]])
        map_text += line.rstrip() + "\n"
    
    print("Vista previa del mapa generado:")
    print(map_text)
    print("...")

def map_indices_to_text(map_indices):
    """Convierte tensor de índices a texto ASCII"""
    lines = []
    for i in range(map_indices.shape[0]):
        line = ''.join([IDX_MAP.get(idx.item(), ' ') for idx in map_indices[i]])
        # Eliminar espacios al final de línea
        lines.append(line.rstrip())
    
    # Eliminar líneas vacías al final
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def enhance_map(map_text):
    """Mejora el mapa generado para garantizar jugabilidad"""
    lines = map_text.split('\n')
    
    # Asegurar 3 líneas vacías al inicio
    while len(lines) < 3 or ''.join(lines[:3]).strip():
        lines.insert(0, '')
    
    # Verificar elementos esenciales
    has_player = JUGADOR in map_text
    has_goal = META in map_text
    has_floor = SUELO in map_text
    
    if not has_player:
        # Añadir jugador en la primera línea no vacía
        for i in range(3, len(lines)):
            if lines[i].strip() and JUGADOR not in lines[i]:
                # Buscar posición sobre un bloque de suelo
                for j in range(i+1, min(i+3, len(lines))):
                    if j < len(lines) and SUELO in lines[j]:
                        idx = lines[j].find(SUELO)
                        if idx >= 0:
                            lines[i] = lines[i][:idx] + JUGADOR + lines[i][idx+1:]
                            has_player = True
                            break
                if has_player:
                    break
        
        # Si no se pudo añadir jugador de forma ideal, añadirlo en primera línea
        if not has_player:
            lines[3] = JUGADOR + (lines[3][1:] if len(lines[3]) > 0 else '')
    
    if not has_goal:
        # Añadir meta al final del mapa
        goal_added = False
        for i in range(4, len(lines)-2):
            if lines[i].strip() and '|>' not in lines[i]:
                # Añadir la meta hacia el final de la línea
                end_pos = len(lines[i].rstrip())
                if end_pos > 0:
                    lines[i] = lines[i][:end_pos] + '  |>'
                    # Añadir soporte debajo de la meta
                    if i+1 < len(lines):
                        while len(lines[i+1]) < end_pos + 4:
                            lines[i+1] += ' '
                        lines[i+1] = lines[i+1][:end_pos] + '  ' + SUELO * 3 + lines[i+1][end_pos+5:] if end_pos+5 < len(lines[i+1]) else lines[i+1][:end_pos] + '  ' + SUELO * 3
                    goal_added = True
                    break
        
        # Si no se pudo añadir meta, hacerlo en una nueva línea
        if not goal_added:
            lines.append('                                 |>')
            lines.append('                                 ' + SUELO * 3)
    
    if not has_floor or map_text.count(SUELO) < 5:
        # Añadir suelo base
        lines.append(SUELO * 50)
    
    # Añadir elementos decorativos si faltan
    if MONEDA not in map_text:
        # Añadir algunas monedas
        for i in range(4, min(15, len(lines))):
            if random.random() < 0.3 and lines[i].strip():
                pos = random.randint(0, max(0, len(lines[i])-1))
                if pos < len(lines[i]) and lines[i][pos] == ' ':
                    lines[i] = lines[i][:pos] + MONEDA + lines[i][pos+1:]
    
    if BLOQUE not in map_text:
        # Añadir algunos bloques sorpresa
        for i in range(4, min(15, len(lines))):
            if random.random() < 0.2 and lines[i].strip():
                pos = random.randint(0, max(0, len(lines[i])-1))
                if pos < len(lines[i]) and lines[i][pos] == ' ':
                    lines[i] = lines[i][:pos] + BLOQUE + lines[i][pos+1:]
    
    # Eliminar líneas vacías al final
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def generate_maps_with_rating(model, diffusion, device, rating_system, 
                           num_maps=NUM_MAPS_GEN, output_dir="mapas_diffusion"):
    """Genera mapas y los evalúa con Gemini"""
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Generando {num_maps} mapas en '{output_dir}'...")
    
    good_maps = 0
    total_score = 0
    
    for i in range(1, num_maps + 1):
        try:
            # Variar la fuerza de guía para generar mapas diversos
            guidance_strength = 1.0 + (i / num_maps) * 3.0  # Entre 1.0 y 4.0
            
            print(f"Generando mapa {i}/{num_maps} (guidance={guidance_strength:.2f})...")
            
            # Generar mapa utilizando el modelo de difusión
            map_indices = generate_map_sample(
                model=model,
                diffusion=diffusion,
                device=device,
                batch_size=1,
                guide_strength=guidance_strength
            )
            
            # Convertir índices a texto
            map_text = map_indices_to_text(map_indices[0])
            
            # Mejorar mapa para garantizar jugabilidad
            map_text = enhance_map(map_text)
            
            # Evaluar con Gemini
            score = rating_system.rate_map(map_content=map_text)
            total_score += score
            
            if score >= QUALITY_THRESHOLD:
                good_maps += 1
                
            print(f"Calidad del mapa: {score}% {'✅' if score >= QUALITY_THRESHOLD else '❌'}")
            
            # Guardar mapa
            filename = f"diffusion_map{i}_score{score}_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(map_text)
            
            print(f"✅ Mapa {i} guardado en: {filepath}")
            
            # Vista previa
            preview_lines = map_text.split('\n')[:8]
            preview = '\n'.join(preview_lines) + ('\n...' if len(map_text.split('\n')) > 8 else '')
            print(f"Vista previa:\n{preview}\n")
            
        except Exception as e:
            print(f"❌ Error generando mapa {i}: {str(e)}")
    
    avg_score = total_score / num_maps
    print(f"\n✨ Resumen de generación:")
    print(f"  - Mapas generados: {num_maps}")
    print(f"  - Mapas con calidad ≥{QUALITY_THRESHOLD}%: {good_maps} ({good_maps/num_maps*100:.1f}%)")
    print(f"  - Calidad media: {avg_score:.1f}%")
    print(f"  - Mapas guardados en: {output_dir}")

def generate_artificial_training_maps(num_maps=20):
    """Genera mapas artificiales básicos pero válidos para entrenamiento"""
    valid_files = []
    os.makedirs("mapas_artificiales", exist_ok=True)
    
    for i in range(1, num_maps + 1):
        template = f"""

J
■■■■  □  C   
      ■■■■    
"""
        # Generar un camino aleatorio
        num_platforms = random.randint(3, 8)
        curr_pos = 10
        line = ""
        
        for _ in range(num_platforms):
            platform_len = random.randint(2, 7)
            gap = random.randint(2, 5)
            line += " " * gap + SUELO * platform_len
            curr_pos += gap + platform_len
            
            # Añadir algunos elementos aleatorios encima de la plataforma
            if random.random() < 0.5:
                top_line = " " * curr_pos
                for j in range(platform_len):
                    if random.random() < 0.3:
                        element = random.choice([MONEDA, BLOQUE, ENEMIGO])
                        curr_idx = curr_pos - platform_len + j
                        if curr_idx < len(top_line):
                            top_line = top_line[:curr_idx] + element + top_line[curr_idx+1:]
                template += top_line + "\n"
        
        # Añadir plataforma final con meta
        template += line + "\n"
        template += " " * (curr_pos - 5) + "|>\n"
        template += " " * (curr_pos - 5) + "■■■■■■"
        
        # Guardar el mapa
        map_file = f"mapas_artificiales/mapa_artificial_{i:02d}.txt"
        with open(map_file, "w", encoding="utf-8") as f:
            f.write(template)
        
        valid_files.append(map_file)
    
    return valid_files

def preprocess_map_files(map_files):
    """Preprocesa y valida los archivos de mapas antes del entrenamiento"""
    valid_files = []
    
    for file in tqdm(map_files, desc="Validando mapas"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar elementos esenciales
            if JUGADOR in content and META in content and SUELO in content:
                valid_files.append(file)
        except Exception as e:
            print(f"Error al procesar {file}: {e}")
    
    print(f"De {len(map_files)} mapas, {len(valid_files)} son válidos para entrenamiento")
    return valid_files

def main():
    print("🎮 Iniciando entrenamiento del modelo de DIFUSIÓN para mapas ASCII")
    print(f"⭐ Objetivo: Generar mapas de plataformas con estructura 2D coherente")
    
    # Configurar semilla para reproducibilidad
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Determinar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Inicializar sistema de evaluación Gemini
    rating_system = MapRatingSystem()
    
    # Cargar archivos de mapas
    map_files = glob.glob(os.path.join(os.getcwd(), "mapa*.txt"))
    map_files = sorted([f for f in map_files if os.path.basename(f)[4:-4].isdigit()])
    
    if not map_files:
        print("⚠️ No se encontraron archivos de mapas válidos")
        return
    
    print(f"Encontrados {len(map_files)} archivos de mapas")
    
    # Filtrar mapas válidos antes de crear el dataset
    valid_map_files = preprocess_map_files(map_files)
    
    if len(valid_map_files) < 10:
        print("⚠️ No hay suficientes mapas válidos para entrenar el modelo")
        print("Generando mapas artificiales para entrenamiento...")
        valid_map_files = generate_artificial_training_maps(20)
    
    # Crear dataset de mapas 2D con solo mapas válidos
    dataset = MapDataset(valid_map_files)
    
    # Dividir en entrenamiento y validación
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Crear dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Inicializar modelo U-Net para la difusión
    model = SimpleUNet(in_channels=len(CHAR_MAP), out_channels=len(CHAR_MAP))
    model.to(device)
    print(f"Modelo inicializado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Inicializar el proceso de difusión
    diffusion = GaussianDiffusion(num_diffusion_steps=DIFFUSION_STEPS, device=device)
    
    # Verificar si hay un modelo pre-entrenado
    model_files = glob.glob("diffusion_model_*.pt")
    if model_files:
        latest_model = max(model_files)
        print(f"Cargando modelo existente: {latest_model}")
        
        checkpoint = torch.load(latest_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado (epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})")
        
        # Preguntar si quiere continuar entrenamiento
        train_more = input("¿Deseas continuar el entrenamiento? (s/n): ").lower() == 's'
        
        if not train_more:
            # Solo generar mapas
            generate_maps_with_rating(model, diffusion, device, rating_system)
            return
    
    # Inicializar optimizador
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenar el modelo
    print("🚀 Iniciando entrenamiento del modelo de difusión...")
    loss_history = train_diffusion_model(
        model=model,
        dataloader=train_dataloader,
        diffusion=diffusion, 
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS
    )
    
    print("\n✅ Entrenamiento completado")
    
    # Guardar modelo final
    final_model_path = "diffusion_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }, final_model_path)
    print(f"Modelo final guardado como {final_model_path}")
    
    # Generar mapas de muestra
    print("\n🎲 Generando mapas finales con evaluación Gemini...")
    generate_maps_with_rating(model, diffusion, device, rating_system)
    
    print("\n✨ ¡Proceso completado! ✨")

if __name__ == "__main__":
    main()