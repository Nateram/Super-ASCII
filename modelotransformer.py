# Nota: Instala las librerías necesarias si no las tienes:
# pip install torch numpy requests tqdm openai google-generativeai

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
import re # Importar regex
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

# === CONFIGURACIÓN DE GEMINI API ===
# Preferible usar variables de entorno: export GEMINI_API_KEY='tu_clave'
# Puedes obtener tu clave aquí: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyCoFRhRa1zamcMIqIxIuBChtehsoRE5AUM" # REEMPLAZAR O USAR ENV VAR
GEMINI_MODEL = "gemini-2.5-flash" # Modelo más reciente y económico
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
# La URL base para Gemini 1.5 Flash
QUALITY_THRESHOLD = 60  # Puntuación mínima para considerar un mapa "bueno" para entrenamiento/guardado final
CONSECUTIVE_MAPS_STOP = 3 # Mapas consecutivos que deben superar el umbral para detener entrenamiento
RATINGS_FILE = "map_ratings_gemini.json"  # Archivo para guardar valoraciones con Gemini

# Constantes para elementos del juego
JUGADOR = 'J'
META_START = '|' # Usamos solo el inicio de la meta
SUELO = '■'
AGUA = '🟦'
MONEDA = 'C'
BLOQUE = '□'
ENEMIGO = '☉'
EMPTY = ' ' # Espacio vacío
NEWLINE = '\n' # Nueva línea

# === VOCABULARIO FIJO ===
# Definimos todos los posibles caracteres que pueden aparecer + tokens especiales
GAME_CHARS = [JUGADOR, META_START, '>', SUELO, AGUA, MONEDA, BLOQUE, ENEMIGO, EMPTY, NEWLINE]
SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']
VOCAB = SPECIAL_TOKENS + GAME_CHARS # El orden importa para los índices

# Constantes para tokenización
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_IDX = VOCAB.index(PAD_TOKEN)
BOS_IDX = VOCAB.index(BOS_TOKEN)
EOS_IDX = VOCAB.index(EOS_TOKEN)
UNK_IDX = VOCAB.index(UNK_TOKEN)
VOCAB_SIZE = len(VOCAB)

# Configuración del modelo Transformer
EMBED_SIZE = 256       # Dimensión de embeddings
HIDDEN_SIZE = 512      # Tamaño de las capas intermedias
NUM_LAYERS = 4         # Reducimos capas para mitigar sobreajuste en datos pequeños
NUM_HEADS = 8          # Cabezas de atención
DROPOUT = 0.1          # Tasa de dropout
BATCH_SIZE = 8         # Tamaño del batch
MAX_LEN = 512          # Reducimos longitud máxima para datos pequeños
EPOCHS = 50            # Aumentamos épocas, entrenamiento más largo
LEARNING_RATE = 3e-4   # Ajustamos LR
NUM_MAPS_GEN = 10      # Mapas a generar por ronda de evaluación/generación final
EVAL_FREQUENCY = 3     # Evaluar cada X épocas

# === JUGABILIDAD BÁSICA (para prompts y post-proceso) ===
MAX_PLAYER_RUN_JUMP_DIST = 6 # Distancia horizontal máxima de salto
MAX_PLAYER_JUMP_HEIGHT = 4  # Altura vertical máxima de salto
MAP_WIDTH = 120
MAP_HEIGHT = 25
class MapRatingSystem:
    """Sistema para valorar mapas usando Gemini API"""
    def __init__(self, ratings_file=RATINGS_FILE):
        self.ratings_file = ratings_file
        self.ratings = self._load_ratings()
        self.gemini_client = self._init_gemini_client()
        self.system_prompt = self._create_system_prompt()
        self.reference_maps = self._load_reference_maps() # Para dar contexto al LLM

    def _load_ratings(self):
        """Cargar valoraciones existentes o crear un nuevo archivo"""
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, 'r', encoding='utf-8') as f:
                    # Cargar y convertir claves de string a int si es necesario (hash es int)
                    data = json.load(f)
                    # Convertir hashes a int si están guardados como string
                    return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()}
            except Exception as e:
                print(f"Error al cargar {self.ratings_file}: {e}, creando nuevo archivo.")
        return {}

    def _load_reference_maps(self):
        """Cargar los primeros 10 mapas originales como referencia para Gemini"""
        maps = []
        # Buscar específicamente los primeros 10 mapas por nombre
        ref_files = [f"mapa{i:02d}.txt" for i in range(1, 11)]
        
        for file_name in ref_files:
             file_path = os.path.join(os.getcwd(), file_name)
             if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        maps.append(f.read())
                except Exception as e:
                    print(f"Error al cargar mapa de referencia {file_name}: {e}")

        if maps:
            print(f"Cargados {len(maps)} mapas de referencia para el LLM.")
        else:
            print("⚠️ No se encontraron mapas de referencia (mapa01.txt a mapa10.txt). La evaluación del LLM puede ser menos específica del estilo.")

        return maps

    def save_ratings(self):
        """Guardar valoraciones a archivo JSON"""
        try:
            with open(self.ratings_file, 'w', encoding='utf-8') as f:
                json.dump(self.ratings, f, indent=2)
        except Exception as e:
            print(f"Error al guardar {self.ratings_file}: {e}")

    def _init_gemini_client(self):
        """Inicializar cliente Gemini API"""
        if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSy...":
            print("ERROR: API Key de Gemini no configurada correctamente.")
            print("Establece la variable de entorno GEMINI_API_KEY o edita el script.")
            return None
        try:
            # Gemini 1.5 Flash usa la API de Vertex AI o Google AI Studio (anteriormente makersuite)
            # Aquí usamos la URL de Google AI Studio V1 Beta
            # from google.generativeai.types import GenerationConfig
            # import google.generativeai as genai
            # genai.configure(api_key=GEMINI_API_KEY)
            # model = genai.GenerativeModel(GEMINI_MODEL)
            # print(f"Gemini client initialized using model: {GEMINI_MODEL}")
            # return model # Retorna el objeto modelo
             
            # Alternativa usando requests directamente si la librería google-generativeai no funciona o no se quiere usar
            print(f"Usando requests para llamar a Gemini API ({GEMINI_MODEL}).")
            return True # Solo indicamos que la clave parece válida y usaremos requests
            
        except Exception as e:
            print(f"Error al inicializar Gemini API: {e}")
            return None # Retorna None si falla

    def _make_gemini_request(self, system_prompt, user_prompt):
        """Hacer petición a la API de Gemini usando requests"""
        if not self.gemini_client: # Si gemini_client es True (usando requests) o un objeto modelo
             print("Cliente Gemini no inicializado. No se puede hacer petición.")
             return None
             
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Estructura de la petición para Gemini 1.5 Flash (requests)
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 100
            },
             # Gemini 1.5 Flash no usa un 'system' role directo en este endpoint.
             # Incluimos el system prompt en el user prompt o en el chat history si fuera más complejo.
             # Para una única petición, lo combinamos en el user_prompt.
        }
        
        full_prompt = system_prompt + "\n\n" + user_prompt # Combinar prompts para el request
        data["contents"][0]["parts"][0]["text"] = full_prompt # Usar el prompt combinado
        
        # Formato de URL para Gemini 1.5 Flash
        url = f"{GEMINI_API_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=45) # Aumentar timeout
            response.raise_for_status() # Lanza HTTPError para respuestas 4xx/5xx
            
            result = response.json()
            
            # Verificar si hay contenido generado
            if 'candidates' in result and len(result['candidates']) > 0 and 'content' in result['candidates'][0]:
                 # Extraer texto de la primera parte del primer candidato
                 if 'parts' in result['candidates'][0]['content'] and len(result['candidates'][0]['content']['parts']) > 0:
                     return result['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Manejar posibles errores o respuestas vacías
            if 'promptFeedback' in result:
                 print(f"Gemini Prompt Feedback: {result['promptFeedback']}")
            
            print(f"Respuesta inesperada o vacía de Gemini: {json.dumps(result, indent=2)}")
            return None

        except requests.exceptions.Timeout:
            print("Error en petición a Gemini: Timeout.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error en petición a Gemini: {e}")
            # print(f"Response body: {e.response.text}") # Opcional: imprimir cuerpo de error
            return None
        except Exception as e:
            print(f"Error procesando respuesta de Gemini: {e}")
            return None

    def _create_system_prompt(self):
        """Crear prompt para Gemini con ejemplos de mapas perfectos"""
        # Nota: Con la API de Gemini 1.5 Flash (AI Studio endpoint), el rol 'system' no es soportado
        # en la estructura de 'contents'. Se incluye en el 'user' prompt o se maneja como un turno inicial
        # en un chat. Para este caso simple, lo incluimos en el user prompt.
        prompt = """Eres un evaluador experto de mapas de plataformas ASCII para un juego tipo Super Mario Bros. Tu única tarea es dar una puntuación del 1 al 100.

Elementos del mapa:
- J: Jugador
- |>: Meta/Bandera
- ■: Bloques/Plataformas/Suelo
- 🟦: Agua
- C: Monedas
- □: Bloques sorpresa
- ☉: Enemigos

Considera los siguientes factores para tu evaluación (puntúa de 1 a 100):
1. Estilo y coherencia visual, similar a los mapas de referencia que te mostraré.
2. Presencia de elementos esenciales (jugador J y meta |>). Un mapa sin J o |> no puede ser 100.
3. Posicionamiento lógico de plataformas y elementos para la jugabilidad (considerando saltos de hasta {MAX_PLAYER_RUN_JUMP_DIST} horizontal y {MAX_PLAYER_JUMP_HEIGHT} vertical).
4. Distribución equilibrada de desafíos y recompensas.
5. **Un mapa muy simple (ej: J, línea recta de bloques, meta) debería tener una valoración aproximada de 50.**

Proporciona SOLO una puntuación numérica entera del 1 al 100.

"""
        return prompt

    def rate_map(self, map_content=None, map_file=None):
        """Evaluar un mapa usando Gemini API comparándolo con los de referencia"""
        if map_content is None and map_file is None:
            raise ValueError("Debes proporcionar contenido de mapa o un archivo")

        if map_content is None:
            map_path = os.path.abspath(map_file)
            try:
                with open(map_path, 'r', encoding='utf-8') as f:
                    map_content = f.read()
            except Exception as e:
                print(f"Error leyendo archivo {map_file}: {e}")
                return 50  # Valor por defecto si falla lectura

        # Limpiar mapa_content de posibles tokens especiales internos del modelo
        map_content = map_content.replace(PAD_TOKEN, '').replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').replace(UNK_TOKEN, '?').strip()
        
        # Verificar si ya tenemos valoración para este mapa
        # Usar solo el contenido significativo para el hash
        map_hash = hash(map_content)
        if map_hash in self.ratings:
            # print(f"DEBUG: Usando valoración cacheada para hash {map_hash}: {self.ratings[map_hash]}")
            return self.ratings[map_hash]

        # Si no hay cliente Gemini o no hay mapas de referencia, usar valoración fallback
        if self.gemini_client is None: # Ahora gemini_client es True o None
            print("⚠️ Cliente Gemini no disponible. Usando valoración fallback.")
            score = self._fallback_rating(map_content)
            self.ratings[map_hash] = score # Usamos el hash int como key
            self.save_ratings()
            return score
        
        # Evaluar con Gemini
        try:
            # Crear prompt con mapas de referencia y el mapa a evaluar
            user_prompt_content = "Aquí están los mapas de referencia:\n\n"
            for i, ref_map in enumerate(self.reference_maps):
                 # Limpiar mapas de referencia de tokens especiales si se cargaron mal
                 ref_map = ref_map.replace(PAD_TOKEN, '').replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').replace(UNK_TOKEN, '?').strip()
                 user_prompt_content += f"MAPA REFERENCIA {i+1}:\n```\n{ref_map}\n```\n\n"

            user_prompt_content += "Ahora, evalúa este nuevo mapa con una puntuación de 1 a 100:\n\n```\n" + map_content + "\n```\nPuntuación:" # Pedir la puntuación explícitamente al final


            response_text = self._make_gemini_request(self.system_prompt, user_prompt_content) # Pasar system y user prompts por separado/combinados

            if response_text is None:
                # Si la API falla, usar valoración fallback
                print("⚠️ Petición a Gemini falló o devolvió respuesta vacía. Usando valoración fallback.")
                score = self._fallback_rating(map_content)
                self.ratings[map_hash] = score
                self.save_ratings()
                return score

            # Intentar extraer un número de la respuesta
            score_match = re.search(r'\b(\d{1,3})\b', response_text)

            if score_match:
                score = int(score_match.group(1))
                score = max(1, min(100, score)) # Validar rango [1, 100]
            else:
                print(f"⚠️ No se pudo extraer puntuación numérica de la evaluación Gemini: '{response_text}'. Usando valoración fallback.")
                score = self._fallback_rating(map_content)

            # Guardar y devolver puntuación
            self.ratings[map_hash] = score
            self.save_ratings()
            # print(f"DEBUG: Gemini rate_map returning score {score}")
            return score

        except Exception as e:
            print(f"Error general en evaluación Gemini: {e}")
            score = self._fallback_rating(map_content)
            self.ratings[map_hash] = score
            self.save_ratings()
            return score

    def _fallback_rating(self, map_content):
        """Método alternativo de puntuación si Gemini falla"""
        # Simple heuristic fallback - Aumento de penalización por elementos faltantes
        
        # Limpieza básica
        map_content = map_content.replace(PAD_TOKEN, '').replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').replace(UNK_TOKEN, '?').strip()

        # Penalizar mapas vacíos o casi vacíos
        if len(map_content.replace('\n', '').strip()) < 10:
            return 1
            
        # Penalizar severamente mapas que son solo líneas de bloques o caracteres repetidos
        lines = map_content.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]
        if len(non_empty_lines) > 0 and all(len(set(line.strip())) <= 1 for line in non_empty_lines if line.strip()):
             return 5

        # Base de puntuación
        score = 30 # Puntuación base

        # Penalizaciones por elementos esenciales faltantes (más severo)
        if JUGADOR not in map_content:
            score -= 30
        if META_START not in map_content: # Buscar '|' como indicador de meta
            score -= 30
        if SUELO not in map_content or map_content.count(SUELO) < 5: # Mínimo de suelo
            score -= 25 # Aumentada penalización
        
        # Verificar estructura mínima (J sobre o cerca de ■)
        player_over_floor = False
        map_lines = map_content.split('\n')
        for y, line in enumerate(map_lines):
            p_indices = [i for i, char in enumerate(line) if char == JUGADOR]
            for p_idx in p_indices:
                 if y + 1 < len(map_lines) and p_idx < len(map_lines[y+1]):
                     if map_lines[y+1][p_idx] in [SUELO, BLOQUE, ENEMIGO]: # Jugador sobre algo sólido
                         player_over_floor = True
                         break
            if player_over_floor: break
        if not player_over_floor and JUGADOR in map_content: # Solo penalizar si hay jugador pero no sobre suelo
             score -= 20 # Penalización por jugador mal posicionado
             
        # Verificar meta alcanzable (muy básico: meta después del jugador horizontalmente)
        player_idx = map_content.find(JUGADOR)
        meta_idx = map_content.find(META_START)
        if player_idx != -1 and meta_idx != -1 and meta_idx < player_idx:
             score -= 15 # Penalizar meta antes del jugador (puede ser injusto con teleports, pero es un fallback)

        # Premiar variedad y cantidad razonable de otros elementos
        score += min(20, map_content.count(MONEDA) * 2) # Max +20 por monedas
        score += min(15, map_content.count(ENEMIGO) * 3) # Max +15 por enemigos (quieren desafío)
        score += min(15, map_content.count(BLOQUE) * 3) # Max +15 por bloques
        # score += min(10, map_content.count(AGUA) * 5) # Opcional: premiar agua si se usa
        
        # Premiar longitud/altura razonable (evitar mapas de 1 línea)
        num_lines = len(non_empty_lines)
        if num_lines > 5: score += min(15, num_lines * 1) # Hasta +15 por altura

        # Asegurar que el puntaje esté en el rango [1, 100]
        return max(1, min(100, score))

    def get_weighted_maps(self, map_files):
        """Obtener mapas con pesos según su calidad y relevancia"""
        weights = []
        paths = []

        # Analizar calidad de mapas de referencia (mapa01.txt - mapa10.txt)
        # Damos peso extra a los que tenemos guardados con buena puntuación por Gemini/Fallback
        
        for map_file in map_files:
            path = os.path.abspath(map_file)
            content = ""
            try:
                with open(map_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                pass # Ignorar si no se puede leer

            map_hash = hash(content)
            score = self.ratings.get(map_hash, self._fallback_rating(content)) # Usar score guardado o fallback
            
            # Asignar peso:
            # Mapas de referencia (mapa01-10) tienen peso base alto + ajuste por score
            # Otros mapas tienen peso base más bajo + ajuste por score
            base_weight = 5.0 # Peso base para la mayoría
            if os.path.basename(map_file).startswith("mapa") and os.path.basename(map_file)[4:-4].isdigit() and int(os.path.basename(map_file)[4:-4]) <= 10:
                 base_weight = 15.0 # Peso base más alto para referencias
            
            # Ajustar peso base por puntuación (lineal o exponencial)
            # Usaremos un ajuste lineal: peso = base + (score/10) * factor
            score_adjustment = (score / 100.0) * 10.0 # Ajuste de hasta +10 basado en score
            
            weight = base_weight + score_adjustment
            
            weights.append(max(1.0, weight)) # Asegurar peso mínimo de 1
            paths.append(path)

        # Normalizar pesos opcionalmente si el sampler lo requiere
        # WeightedRandomSampler no necesita normalización, solo los pesos relativos
            
        return paths, weights

class MapTokenizer:
    def __init__(self):
        self.char2idx = {char: idx for idx, char in enumerate(VOCAB)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(VOCAB)
        self.pad_idx = PAD_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX # En un vocabulario fijo, unk es para chars no esperados
    
    def encode(self, text):
        """Convierte texto a IDs de tokens, manejando UNK y truncando/paddeando"""
        # Asegurar longitud máxima, añadir BOS y EOS
        tokens = [self.char2idx.get(char, self.unk_idx) for char in text]
        
        # **VALIDACIÓN: Verificar que todos los tokens estén en rango**
        for i, token in enumerate(tokens):
            if token < 0 or token >= self.vocab_size:
                print(f"⚠️ ERROR en encode: Token fuera de rango en posición {i}: {token} (vocab_size: {self.vocab_size})")
                tokens[i] = self.unk_idx  # Corregir con UNK
        
        # Añadir BOS y EOS
        tokens = [self.bos_idx] + tokens + [self.eos_idx]
        
        # Truncar si excede MAX_LEN
        if len(tokens) > MAX_LEN:
             tokens = tokens[:MAX_LEN]
        
        return tokens # No paddeamos aquí, lo hará el collate_batch

    def decode(self, ids, skip_special=True):
        """Convierte IDs de tokens a texto"""
        chars = []
        for idx in ids:
            # Asegurar que idx está dentro del rango del vocabulario
            if idx < 0 or idx >= self.vocab_size:
                 char = self.unk_token
            else:
                 char = self.idx2char[idx]
            
            if skip_special and char in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
                 continue
            chars.append(char)
            
        return ''.join(chars)

class MapDataset(Dataset):
    """Dataset básico para mapas tokenizados"""
    def __init__(self, tokenizer, files):
        self.examples = []
        self.tokenizer = tokenizer
        
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    # Preprocesamiento: asegurar formato consistente
                    content = self.preprocess_map(content)
                    # Tokenizar el mapa
                    tokens = self.tokenizer.encode(content)
                    # Solo añadir si el mapa no es casi vacío después del procesamiento
                    if len(tokens) > 5: # Mínimo de tokens significativo
                         self.examples.append(tokens)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Loaded {len(self.examples)} maps for dataset")

    def preprocess_map(self, map_text):
        """Normaliza el formato del mapa"""
        # Eliminar BOM si existe
        if map_text.startswith('\ufeff'):
             map_text = map_text[1:]
             
        # Reemplazar caracteres extraños por UNK si no están en vocabulario fijo
        processed_text = "".join([char if char in self.tokenizer.char2idx else UNK_TOKEN for char in map_text])
        
        # Asegurar al menos 3 líneas (las vacías iniciales + algo)
        lines = processed_text.split('\n')
        while len(lines) < 3:
             lines.insert(0, '')
        # Asegurar 3 líneas vacías al inicio (esto es una convención de los mapas de ejemplo)
        # No *insertar* si ya hay, solo asegurar que las 3 primeras no tienen contenido significativo
        if len(lines) > 0 and lines[0].strip(): lines.insert(0, '')
        if len(lines) > 1 and lines[1].strip(): lines.insert(1, '')
        if len(lines) > 2 and lines[2].strip(): lines.insert(2, '')
        
        # Unir y devolver
        return '\n'.join(lines)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

class WeightedMapDataset(MapDataset): # Heredamos de MapDataset
    """Dataset que pondera mapas según su calidad"""
    def __init__(self, tokenizer, files, weights):
        super().__init__(tokenizer, files) # Llama al init de MapDataset para cargar y procesar
        
        # Asegurarse de que el número de pesos coincide con el número de ejemplos cargados válidos
        if len(self.examples) != len(weights):
            print(f"⚠️ Warning: Number of examples ({len(self.examples)}) does not match number of weights ({len(weights)}). Using uniform weights.")
            self.dataset_weights = [1.0] * len(self.examples)
        else:
            self.dataset_weights = weights

        # Asegurar que los pesos son flotantes y positivos
        self.dataset_weights = [max(0.1, float(w)) for w in self.dataset_weights] # Peso mínimo para evitar 0
        
        print(f"Loaded {len(self.examples)} maps with weighted sampling enabled")
    
    def get_sampler(self):
        """Crea un sampler ponderado basado en calidad de mapas"""
        if not self.dataset_weights:
             print("⚠️ No hay pesos disponibles. Usando sampler aleatorio simple.")
             return random_split(self, [len(self)])[0] # Truco para obtener un subset aleatorio
             
        # WeightedRandomSampler requiere que sum(weights) > 0
        total_weight = sum(self.dataset_weights)
        if total_weight <= 0:
             print("⚠️ Suma total de pesos es cero o negativa. Usando sampler aleatorio simple.")
             return random_split(self, [len(self)])[0]
             
        return WeightedRandomSampler(
            weights=self.dataset_weights,
            num_samples=len(self.examples) * 10, # Muestrear muchas veces para que se vea el efecto del peso
            replacement=True # Permite muestrear el mismo ejemplo varias veces
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # .transpose(0, 1) # Shape [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe shape: [1, max_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MapTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_SIZE, nhead=NUM_HEADS, 
                 num_layers=NUM_LAYERS, dim_feedforward=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX) # Añadir padding_idx
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=MAX_LEN)
        
        # Usar TransformerDecoderLayer para generación secuencial
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Importante para que el batch sea la primera dimensión
        )
        
        # TransformerDecoder necesita un Encoder dummy o entrenarse de forma no estándar
        # Un enfoque más simple para generación secuencial (como GPT) es usar TransformerEncoder con máscara causal
        # Volvemos al TransformerEncoder con máscara causal
        encoder_layer = nn.TransformerEncoderLayer(
             d_model=d_model,
             nhead=nhead,
             dim_feedforward=dim_feedforward,
             dropout=dropout,
             batch_first=True
         )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers) # Cambiado a self.transformer
        
        self.output_layer = nn.Linear(d_model, vocab_size) # Cambiado a self.output_layer
        
        self.init_weights()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        # Los pesos del Transformer se inicializan por defecto o por xavier en TransformerEncoderLayer

    def generate_causal_mask(self, sz, device):
        # Máscara causal para TransformerEncoder: cada posición solo atiende a sí misma y a posiciones anteriores
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len]
        # src_key_padding_mask: [batch_size, seq_len] (True si padding, False si no)

        src = self.embedding(src) * np.sqrt(self.d_model) # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src) # [batch_size, seq_len, d_model]

        # Máscara causal
        seq_len = src.size(1)
        causal_mask = self.generate_causal_mask(seq_len, src.device)

        # Pasar por Transformer Encoder
        # src_key_padding_mask: Máscara para evitar que el modelo atienda a tokens PAD
        output = self.transformer(
            src,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask # Pasar la máscara de padding
        ) # [batch_size, seq_len, d_model]

        # Proyectar a logits de vocabulario
        output = self.output_layer(output) # [batch_size, seq_len, vocab_size]

        return output

def collate_batch(batch, pad_idx):
    # Encontrar longitud máxima en el batch
    max_len = min(MAX_LEN, max(len(item) for item in batch)) # Asegurar que no exceda MAX_LEN

    # Crear batch paddeado y máscara de padding
    padded_batch = []
    padding_mask = [] # True para PAD tokens
    for item in batch:
        # Truncar si es demasiado largo
        item = item.tolist()[:max_len]
        
        # Padding a la derecha hasta max_len
        current_len = len(item)
        padded = item + [pad_idx] * (max_len - current_len)
        mask = [False] * current_len + [True] * (max_len - current_len) # False donde hay datos, True donde hay padding
        
        padded_batch.append(padded)
        padding_mask.append(mask)

    # Convertir a tensores
    return (
        torch.tensor(padded_batch, dtype=torch.long),
        torch.tensor(padding_mask, dtype=torch.bool) # Máscara debe ser bool
    )


def adaptive_temperature(recent_qualities):
    """Adapta la temperatura según la calidad media reciente"""
    if not recent_qualities:
        return 1.5 # Temperatura alta si no hay datos de calidad

    avg_quality = sum(recent_qualities) / len(recent_qualities)
    
    # Escala lineal o sigmoide para mapear calidad (ej. 30-80) a temperatura (ej. 1.8 - 0.8)
    # Queremos alta T para baja calidad (explorar) y baja T para alta calidad (refinar)
    
    # Mapeo simple: T = max_T - (avg_quality / 100) * (max_T - min_T)
    min_T = 0.8 # Temperatura mínima (menos aleatoriedad)
    max_T = 1.8 # Temperatura máxima (más aleatoriedad)
    
    temperature = max_T - (avg_quality / 100.0) * (max_T - min_T)
    
    return max(min_T, min(max_T, temperature)) # Asegurar rango

def generate_map(model, tokenizer, device, max_len=MAX_LEN,
                temperature=1.0, top_k=50, top_p=0.95,
                starter_text=None): # Permitir starter_text opcional
    """
    Genera un nuevo mapa usando el modelo transformer entrenado.
    """
    model = model.to(device)
    model.eval()
    
    # **DEBUG: Verificar tamaños del modelo**
    print(f"DEBUG: vocab_size del tokenizer: {tokenizer.vocab_size}")
    print(f"DEBUG: vocab_size del modelo: {model.vocab_size}")
    if hasattr(model.output_layer, 'out_features'):
        print(f"DEBUG: output_layer out_features: {model.output_layer.out_features}")
    
    # Usar starter_text si se proporciona, de lo contrario usar un inicio base
    if starter_text is None:
        # Estructura inicial MUCHO MÁS DEFINIDA y válida
        starter_text = f"""


{JUGADOR}
{SUELO}{SUELO}{SUELO}{SUELO}{SUELO}  {BLOQUE}  {MONEDA}""" # Un buen punto de partida con J sobre suelo
        
    # Convertir la estructura a tokens
    starter_tokens = tokenizer.encode(starter_text)[:-1] # Excluir EOS del starter
    current_ids = torch.tensor([starter_tokens], dtype=torch.long).to(device)
    
    # **DEBUG: Verificar tokens del starter**
    print(f"DEBUG: starter_tokens: {starter_tokens}")
    print(f"DEBUG: max token in starter: {max(starter_tokens) if starter_tokens else 'N/A'}")
    print(f"DEBUG: current_ids shape: {current_ids.shape}")
    
    # Generación token por token
    with torch.no_grad():
        # Empezar DESPUÉS del starter_text
        for i in range(current_ids.size(1), max_len):
            
            # Predecir el siguiente token
            # El modelo ve current_ids y predice la distribución para *cada* posición,
            # pero solo nos interesa la predicción para el *último* token generado.
            outputs = model(current_ids) # [batch_size, current_seq_len, vocab_size]
            
            # **DEBUG: Verificar shapes**
            if i == current_ids.size(1):  # Solo en la primera iteración
                print(f"DEBUG: outputs shape: {outputs.shape}")
                print(f"DEBUG: expected vocab_size: {tokenizer.vocab_size}")
            
            # Tomar los logits solo para la última posición
            next_token_logits = outputs[:, -1, :] / temperature # [batch_size, vocab_size]
            
            # **VALIDACIÓN CRÍTICA: Asegurar que los logits tienen el tamaño correcto**
            if next_token_logits.size(-1) != tokenizer.vocab_size:
                print(f"⚠️ ERROR: Tamaño de logits ({next_token_logits.size(-1)}) != vocab_size ({tokenizer.vocab_size})")
                break
            
            # Filtrar tokens no deseados (padding, unk) - VALIDAR ÍNDICES
            if tokenizer.pad_idx < next_token_logits.size(-1):
                next_token_logits[:, tokenizer.pad_idx] = -float('inf')
            if tokenizer.unk_idx < next_token_logits.size(-1):
                next_token_logits[:, tokenizer.unk_idx] = -float('inf')
            
            # --- Aplicar forzado/guía suave (menos hacks que antes) ---
            current_text_so_far = tokenizer.decode(current_ids[0].tolist(), skip_special=True)

            # Fomentar newlines si la línea actual es larga
            if current_text_so_far.count(NEWLINE) > 0:
                 last_newline_idx = current_text_so_far.rfind(NEWLINE)
                 current_line_len = len(current_text_so_far) - last_newline_idx - 1
                 if current_line_len > MAP_WIDTH * 0.8: # Si la línea es ~80% del ancho
                     newline_idx = tokenizer.char2idx.get(NEWLINE, -1)
                     if newline_idx != -1:
                         next_token_logits[:, newline_idx] += (current_line_len - MAP_WIDTH * 0.8) * 0.5 # Aumentar probabilidad gradualmente

            # Fomentar meta hacia el final del mapa si aún no hay
            if META_START not in current_text_so_far and i > max_len * 0.7:
                 meta_idx = tokenizer.char2idx.get(META_START, -1)
                 gt_idx = tokenizer.char2idx.get('>', -1)
                 if meta_idx != -1 and gt_idx != -1:
                      # Aumentar probabilidad de meta al final
                      next_token_logits[:, meta_idx] += (i - max_len * 0.7) * 0.2 # Aumentar gradualmente
                      next_token_logits[:, gt_idx] += (i - max_len * 0.7) * 0.2


            # --- Sampling ---
            # Top-K filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                # Obtener los top_k valores y sus índices
                values, indices = torch.topk(next_token_logits, top_k)
                # Crear una máscara donde los valores no top-k son infinitos negativos
                next_token_logits[next_token_logits < values[:, -1].unsqueeze(1)] = float('-inf')
            
            # Nucleus (top-p) sampling
            if top_p > 0.0:
                # Aplicar top-p solo a los logits que no fueron filtrados por top-k
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remover tokens con probabilidad acumulada > top_p (y todos después)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift para mantener al menos un token (el primero)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False # Asegurar que el primer elemento nunca se elimina
                
                # Aplicar filtro de top-p a los logits originales (después de top-k)
                # sorted_indices_to_remove tiene la misma forma que sorted_indices
                # Tenemos que mapear de vuelta a los índices originales
                next_token_logits[torch.gather(sorted_indices, -1, sorted_indices_to_remove.long())] = float('-inf')

            # Muestrear próximo token
            probs = torch.softmax(next_token_logits, dim=-1)
            # Manejar el caso donde todos los logits son -inf (probs sumarán 0 o NaN)
            if probs.sum() == 0 or torch.isnan(probs).any():
                 # Fallback: generar un espacio o unk si no se puede generar nada
                 next_token = torch.tensor([[tokenizer.char2idx.get(EMPTY, tokenizer.unk_idx)]], device=device)
            else:
                 next_token = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
                 
            # **VALIDACIÓN CRÍTICA: Verificar que el token generado está en rango válido**
            token_value = next_token.item()
            if token_value < 0 or token_value >= tokenizer.vocab_size:
                print(f"⚠️ ERROR: Token generado fuera de rango: {token_value} (vocab_size: {tokenizer.vocab_size})")
                # Usar token de espacio como fallback seguro
                next_token = torch.tensor([[tokenizer.char2idx.get(EMPTY, tokenizer.unk_idx)]], device=device)


            # Agregar token a la secuencia
            current_ids = torch.cat([current_ids, next_token], dim=1) # [batch_size, current_seq_len + 1]
            
            # Si generamos token de fin, terminamos (después de agregarlo)
            if next_token.item() == tokenizer.eos_idx:
                break
            
    # Decodificar resultado
    map_ids = current_ids[0].tolist()
    # Eliminar <bos> del inicio si está presente
    if map_ids and map_ids[0] == tokenizer.bos_idx:
        map_ids = map_ids[1:]
    
    map_text = tokenizer.decode(map_ids, skip_special=False) # No saltar special tokens aquí, los manejamos en post-proceso
    
    # Postprocesar para garantizar un mapa válido y limpiar special tokens
    map_text = post_process_map(map_text)
    
    return map_text

def post_process_map(map_text):
    """Corrige posibles problemas estructurales y de tokens especiales en el mapa generado"""
    
    # 1. Limpiar tokens especiales (PAD, BOS, EOS, UNK)
    # Reemplazamos UNK por espacio o un caracter seguro
    cleaned_text = map_text.replace(PAD_TOKEN, '').replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').replace(UNK_TOKEN, ' ')
    
    # 2. Asegurar formato de líneas
    lines = cleaned_text.split(NEWLINE)
    
    # Rellenar líneas al ancho máximo si es necesario
    processed_lines = [line.ljust(MAX_LEN) for line in lines] # Rellenar todas las líneas
    
    # Asegurar altura mínima y 3 líneas iniciales (convención de juego)
    min_total_lines = 15 # Altura mínima razonable
    while len(processed_lines) < min_total_lines:
         processed_lines.append(EMPTY * MAX_LEN) # Añadir líneas vacías paddeadas
         
    # Asegurar 3 líneas vacías al inicio (no insertando si ya existen, sino asegurando que las 3 primeras estén vacías)
    for i in range(min(3, len(processed_lines))):
        if processed_lines[i].strip():
            processed_lines.insert(i, EMPTY * MAX_LEN) # Insertar línea vacía paddeada
            processed_lines.pop() # Eliminar la última para mantener el tamaño
            
    # Asegurar que el mapa no exceda una altura máxima razonable (opcional, si MAX_LEN/WIDTH permite mapas muy grandes)
    # Por ahora MAX_LEN ya limita el texto, pero si se generaron muchisimos newlines, podríamos truncar.
    # MAX_VERTICAL_LINES = 30 # Ejemplo de límite vertical si MAX_LEN lo permite
    # processed_lines = processed_lines[:MAX_VERTICAL_LINES]

    # 3. Verificar y garantizar elementos esenciales (J, |>, ■)
    final_map_text = NEWLINE.join(processed_lines)
    
    has_player = JUGADOR in final_map_text
    has_goal = f"{META_START}>" in final_map_text # Buscar "|>" completo
    has_floor = SUELO in final_map_text and final_map_text.count(SUELO) > 5 # Mínimo de suelo
    
    final_lines = processed_lines # Trabajar con la lista de líneas procesadas
    
    # Si no hay jugador, añadir uno en una posición plausible
    if not has_player:
        placed = False
        # Buscar la primera línea no vacía (después de las 3 iniciales)
        for y in range(3, len(final_lines) - 1):
             if final_lines[y].strip(): # Si la línea tiene contenido
                 # Buscar un lugar con suelo debajo
                 for x in range(min(20, len(final_lines[y].rstrip()))): # Buscar cerca del inicio
                     if x < len(final_lines[y+1]) and final_lines[y+1][x] in [SUELO, BLOQUE]:
                          final_lines[y] = final_lines[y][:x] + JUGADOR + final_lines[y][x+1:]
                          placed = True
                          break
             if placed: break
        # Si no se pudo colocar sobre suelo, forzar en una posición por defecto
        if not placed:
             final_lines[3] = final_lines[3][:5] + JUGADOR + final_lines[3][6:] # Poner en (5, 3)
             
    # Si no hay meta, añadir una en una posición plausible
    if not has_goal:
        placed = False
         # Buscar la última línea con contenido significativa hacia el final del mapa
        for y in range(len(final_lines) - 5, 5, -1): # Buscar de abajo hacia arriba, evitando bordes
             if final_lines[y].strip() and f"{META_START}>" not in final_lines[y]:
                  # Buscar una posición hacia el final de la línea, idealmente con suelo debajo
                  line_content_end_x = len(final_lines[y].rstrip())
                  if line_content_end_x > MAX_LEN * 0.6: # Solo si la línea tiene contenido suficientemente a la derecha
                       # Buscar suelo debajo en esa zona
                       for x_offset in range(min(10, MAX_LEN - line_content_end_x)): # Buscar cerca del final del contenido
                            test_x = line_content_end_x + x_offset
                            if test_x < MAX_LEN - 1 and y + 1 < len(final_lines) and final_lines[y+1][test_x] in [SUELO, BLOQUE]:
                                 # Colocar meta |> a 1 unidad horizontal antes
                                 meta_x = max(0, test_x - 1)
                                 # Asegurarse de que hay espacio
                                 if final_lines[y][meta_x] == EMPTY and final_lines[y][meta_x+1] == EMPTY:
                                      final_lines[y] = final_lines[y][:meta_x] + f"{META_START}>" + final_lines[y][meta_x+2:]
                                      placed = True
                                      break
                       if placed: break
        
        # Si no se pudo colocar sobre suelo, forzar al final de una línea
        if not placed:
             target_y = max(3, len(final_lines) - 5) # Ponerlo en una línea cerca del final
             final_lines[target_y] = final_lines[target_y].rstrip() + EMPTY*5 + f"{META_START}>" # Forzar al final de la línea
             # Añadir suelo debajo si es posible
             if target_y + 1 < len(final_lines):
                  meta_start_x = final_lines[target_y].rfind(META_START)
                  if meta_start_x != -1:
                       if len(final_lines[target_y+1]) <= meta_start_x:
                            final_lines[target_y+1] = final_lines[target_y+1] + EMPTY * (meta_start_x - len(final_lines[target_y+1])) + SUELO * 3
                       else:
                            final_lines[target_y+1] = final_lines[target_y+1][:meta_start_x] + SUELO * 3 + final_lines[target_y+1][meta_start_x+3:]


    # Garantizar suelo mínimo si no hay suficiente
    if not has_floor:
        # Añadir una línea de suelo en la parte inferior
        final_lines[-1] = SUELO * MAX_LEN # Asegurar la última línea de suelo


    # Unir las líneas finales
    return NEWLINE.join(final_lines)


def safe_generate_map(model, tokenizer, device, max_len=MAX_LEN,
                     temperature=1.0, top_k=50, top_p=0.95,
                     starter_text=None):
    """
    Wrapper seguro para generar mapas que maneja errores de CUDA
    """
    try:
        # Limpiar cache de CUDA antes de generar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return generate_map(model, tokenizer, device, max_len, temperature, top_k, top_p, starter_text)
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            print(f"⚠️ Error de CUDA detectado: {str(e)[:100]}...")
            print("Intentando cambiar a CPU...")
            
            # Intentar mover todo a CPU y generar
            try:
                model_cpu = model.cpu()
                device_cpu = torch.device('cpu')
                
                # Limpiar cache de CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                result = generate_map(model_cpu, tokenizer, device_cpu, max_len, temperature, top_k, top_p, starter_text)
                
                # Devolver modelo a CUDA si es posible
                model.to(device)
                return result
                
            except Exception as e2:
                print(f"❌ Error también en CPU: {e2}")
                return None
        else:
            print(f"❌ Error no relacionado con CUDA: {e}")
            return None
    except Exception as e:
        print(f"❌ Error general durante generación: {e}")
        return None


def run_validation(model, val_dataloader, criterion, device, tokenizer):
    """Ejecuta una pasada de validación y devuelve la pérdida media"""
    model.eval()
    val_losses = []
    val_bar = tqdm(val_dataloader, desc=f"Validación")
    
    with torch.no_grad():
        for inputs, attention_mask in val_bar: # Recibir máscara de padding
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            # Crear targets desplazados (predecir token siguiente)
            # input: [batch, seq_len], target: [batch, seq_len] (desplazado)
            targets = inputs[:, 1:].contiguous() # El target para inputs[i] es inputs[i+1]
            inputs = inputs[:, :-1].contiguous() # El último token del input no tiene target

            # Ajustar máscara de padding para que coincida con inputs/targets
            padding_mask_inputs = attention_mask[:, :-1].to(device) # Máscara para inputs
            
            # Forward pass - pasar máscara de padding
            outputs = model(inputs, src_key_padding_mask=padding_mask_inputs) # [batch, seq_len-1, vocab_size]

            # Calcular pérdida, ignorando tokens PAD en targets
            # Flatten outputs and targets for CrossEntropyLoss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            val_losses.append(loss.item())
            val_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    avg_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
    return avg_loss


def train_model(model, train_dataloader, val_dataloader, tokenizer,
                device, epochs=EPOCHS, lr=LEARNING_RATE, rating_system=None):
    """Entrena el modelo con refuerzo activo de Gemini"""
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # Buffer para ejemplos de alta calidad generados (para entrenamiento extra)
    high_quality_generated_tokens = [] # Almacena tokens de mapas con puntuación alta
    
    # Historial de calidad de generaciones recientes para temperatura adaptativa
    recent_qualities = []
    consecutive_good_maps = 0 # Contador para early stopping

    # Directorio para mapas de entrenamiento (generados durante el entrenamiento)
    train_maps_dir = "mapas_entrenamiento_transformer"
    os.makedirs(train_maps_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # ===== ENTRENAMIENTO ESTÁNDAR (en datos originales + buffer de alta calidad) =====
        model.train()
        train_losses = []
        
        # Combinar dataloader original y ejemplos de alta calidad si los hay
        combined_dataloader = train_dataloader
        # Nota: Integrar high_quality_generated_tokens en el dataloader ponderado sería ideal,
        # pero es complejo con WeightedRandomSampler. Simplificamos entrenando extra.

        train_bar = tqdm(combined_dataloader, desc=f"Entrenamiento estándar")

        for inputs, attention_mask in train_bar: # Recibir máscara de padding
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            # Crear targets desplazados
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()
            
            padding_mask_inputs = attention_mask[:, :-1].to(device) # Máscara para inputs

            outputs = model(inputs, src_key_padding_mask=padding_mask_inputs) # Pasar máscara

            # Calcular pérdida, ignorando padding
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                           targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Aumentamos clip norm un poco
            optimizer.step()

            train_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = sum(train_losses)/len(train_losses)

        # ===== ENTRENAMIENTO ADICIONAL EN EJEMPLOS DE ALTA CALIDAD GENERADOS =====
        if high_quality_generated_tokens:
            print(f"✨ Entrenando en {len(high_quality_generated_tokens)} ejemplos de alta calidad generados...")
            # Convertir lista de tensores a un solo tensor para dataloader
            hq_tensors = torch.stack(high_quality_generated_tokens).to(device)
            hq_dataset = torch.utils.data.TensorDataset(hq_tensors)
            hq_dataloader = torch.utils.data.DataLoader(hq_dataset, batch_size=BATCH_SIZE // 2, shuffle=True) # Usar batch size más pequeño
            
            extra_epochs = 5 # Pocas épocas extra en estos ejemplos
            
            model.train()
            hq_losses = []
            hq_bar = tqdm(range(extra_epochs), desc="Entrenamiento HQ")
            for extra_epoch in hq_bar:
                epoch_hq_losses = []
                for (batch,) in hq_dataloader:
                    inputs = batch[:, :-1].contiguous()
                    targets = batch[:, 1:].contiguous()
                    
                    # Crear máscara de padding para estos ejemplos (asumiendo padding al final)
                    hq_padding_mask_inputs = (inputs == tokenizer.pad_idx).to(device)
                    
                    outputs = model(inputs, src_key_padding_mask=hq_padding_mask_inputs)

                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    
                    # Reducir un poco la pérdida para darles más "importancia" implícita
                    weighted_loss = loss * 0.8 # 20% menos penalización
                    
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_hq_losses.append(loss.item())
                    
                if epoch_hq_losses: hq_losses.append(sum(epoch_hq_losses)/len(epoch_hq_losses))
                hq_bar.set_postfix(loss=f"{hq_losses[-1]:.4f}" if hq_losses else "N/A")

            print(f"✨ Entrenamiento HQ finalizado. Pérdida media HQ: {sum(hq_losses)/len(hq_losses):.4f}" if hq_losses else "No hubo pérdida HQ.")


        # ===== VALIDACIÓN =====
        avg_val_loss = run_validation(model, val_dataloader, criterion, device, tokenizer)
        print(f"Epoch {epoch} - Pérdida train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        # ===== EVALUACIÓN Y REFUERZO CON GEMINI =====
        if epoch % EVAL_FREQUENCY == 0:
            if rating_system.gemini_client is None:
                 print("⚠️ Cliente Gemini no disponible. Saltando evaluación/refuerzo en esta época.")
            else:
                print("\n🔍 Generando mapas para evaluación y refuerzo con Gemini...")

                generated_maps_this_round = []
                scores_this_round = []

                # Calcular temperatura adaptativa
                temperature = adaptive_temperature(recent_qualities)
                print(f"  Temperatura de generación para esta ronda: {temperature:.2f}")

                # Generar N mapas para evaluar
                for i in range(NUM_MAPS_GEN): # Usamos NUM_MAPS_GEN de la config
                    try:
                        # Generar mapa
                        map_text = safe_generate_map(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            max_len=MAX_LEN,
                            temperature=temperature,
                            top_k=50, # Ajustado para más diversidad
                            top_p=0.95 # Ajustado para más diversidad
                        )

                        # Evaluar con Gemini
                        score = rating_system.rate_map(map_content=map_text)
                        scores_this_round.append(score)
                        generated_maps_this_round.append(map_text) # Guardar texto también

                        # Registrar calidad reciente para temp adaptativa
                        recent_qualities.append(score)
                        if len(recent_qualities) > NUM_MAPS_GEN: # Mantener un historial limitado
                            recent_qualities.pop(0)

                        # Guardar mapa generado en carpeta de entrenamiento
                        filename = f"gen_e{epoch}_i{i+1}_score{score}.txt"
                        filepath = os.path.join(train_maps_dir, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(map_text)

                        print(f"  Mapa Generado {i+1}/{NUM_MAPS_GEN}: Calidad {score}%")
                        # print preview? (optional, can clutter console)
                        # print_sample_map(tokenizer, map_text) # Need to adjust print_sample_map

                        # Si el mapa es de buena calidad, añadir a buffer HQ para entrenamiento extra
                        if score >= QUALITY_THRESHOLD:
                             print(f"  ⭐ Mapa {i+1} ({score}%) es de buena calidad. Añadiendo a buffer HQ.")
                             tokens = tokenizer.encode(map_text)
                             # Asegurarse de que el tensor tiene MAX_LEN para apilar (paddeamos en collate_batch)
                             # Aquí solo guardamos la lista de ids, la paddeamos al crear el tensor para el dataloader
                             high_quality_generated_tokens.append(torch.tensor(tokens, dtype=torch.long)) # Guardamos el tensor sin padding aún
                             # Limitar tamaño del buffer HQ
                             if len(high_quality_generated_tokens) > NUM_MAPS_GEN * 2: # Ejemplo: Buffer de tamaño doble de NUM_MAPS_GEN
                                  # Eliminar aleatoriamente el más antiguo o uno de baja puntuación si tuviéramos scores aquí
                                  high_quality_generated_tokens.pop(0) # Simplemente eliminar el más antiguo por ahora

                    except Exception as e:
                        print(f"❌ Error durante generación/evaluación del mapa {i+1}: {e}")

                # Verificar condición de parada temprana (N mapas consecutivos buenos)
                if len(scores_this_round) >= CONSECUTIVE_MAPS_STOP and all(score >= QUALITY_THRESHOLD for score in scores_this_round[-CONSECUTIVE_MAPS_STOP:]):
                    print(f"\n🎉 ¡Condición de parada temprana alcanzada! {CONSECUTIVE_MAPS_STOP} mapas consecutivos con calidad >{QUALITY_THRESHOLD}%")

                    # Guardar modelo especial
                    final_model_path = f'model_quality>{QUALITY_THRESHOLD}_{epoch}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'recent_qualities': recent_qualities,
                        'avg_quality_round': sum(scores_this_round)/len(scores_this_round) if scores_this_round else 0,
                    }, final_model_path)

                    print(f"✓ Modelo final guardado como '{final_model_path}'")
                    # Guardar el mejor modelo general también por si acaso
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss # Guardar la última val loss conocida
                    }, 'best_map_transformer.pt')

                    return # Terminar entrenamiento

        # Guardar el mejor modelo hasta ahora (basado en la última validación)
        # Podríamos guardar basado en la mejor validation loss en lugar de cada vez,
        # pero para asegurar que se guarda, lo hacemos cada época.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss # Guardar la última val loss conocida
        }, 'best_map_transformer.pt')

    print("✅ Entrenamiento completado (se alcanzó el máximo de épocas)")

    # Guardar el mejor modelo general al final si no hubo early stopping
    if not os.path.exists(f'model_quality>{QUALITY_THRESHOLD}_{EPOCHS}.pt'): # Evitar sobrescribir si early stop ocurrió exactamente en la última época
         final_model_path = f'model_finished_e{EPOCHS}.pt'
         torch.save({
             'epoch': EPOCHS,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'val_loss': avg_val_loss
         }, final_model_path)
         print(f"✓ Modelo final de entrenamiento guardado como '{final_model_path}'")


def print_sample_map(tokenizer, map_text):
    """Imprime una vista previa del mapa generado"""
    lines = map_text.split(NEWLINE)
    preview_lines = lines[:min(len(lines), 10)] # Imprime solo las primeras 10 líneas
    
    print("Vista previa del mapa generado:")
    print(NEWLINE.join(preview_lines))
    if len(lines) > 10:
         print("...")

def generate_maps_for_output(model, tokenizer, device, rating_system,
                            num_maps=NUM_MAPS_GEN, output_dir="mapas_transformer_generados"):
    """
    Genera mapas finales usando el modelo entrenado y los evalúa con Gemini.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"Generando {num_maps} mapas finales en '{output_dir}'...")

    good_maps = 0
    total_score = 0
    recent_qualities = []

    model.eval() # Asegurar que el modelo está en modo evaluación

    for i in range(1, num_maps + 1):
        try:
            # Calcular temperatura adaptativa (usando calidad reciente del buffer)
            temperature = adaptive_temperature(recent_qualities) + (i / num_maps) * 0.2 # Pequeña variación
            
            print(f"Generando mapa final {i}/{num_maps} (temperatura={temperature:.2f})...")

            map_text = safe_generate_map(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_len=MAX_LEN,
                temperature=temperature,
                top_k=50,
                top_p=0.95
            )

            # Evaluar con Gemini
            score = rating_system.rate_map(map_content=map_text)
            total_score += score
            recent_qualities.append(score)
            if len(recent_qualities) > NUM_MAPS_GEN:
                recent_qualities.pop(0)

            if score >= QUALITY_THRESHOLD:
                good_maps += 1

            print(f"Calidad del mapa: {score}% {'✅' if score >= QUALITY_THRESHOLD else '❌'}")

            # Guardar mapa
            filename = f"generated_map_{i}_score{score}_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(map_text)

            print(f"✅ Mapa {i} guardado en: {filepath}")

            # Vista previa
            print_sample_map(tokenizer, map_text)


        except Exception as e:
            print(f"❌ Error generando mapa final {i}: {str(e)}")

    avg_score = total_score / num_maps if num_maps > 0 else 0
    print(f"\n✨ Resumen de generación final:")
    print(f"  - Mapas generados: {num_maps}")
    print(f"  - Mapas con calidad ≥{QUALITY_THRESHOLD}%: {good_maps} ({good_maps/num_maps*100:.1f}%)")
    print(f"  - Calidad media: {avg_score:.1f}%")
    print(f"  - Mapas guardados en: {output_dir}")


def main():
    print("🎮 Iniciando entrenamiento/generación con modelo Transformer + Gemini")

    # Configurar semilla para reproducibilidad
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Determinar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar sistema de evaluación Gemini
    rating_system = MapRatingSystem()
    # Verificar si el cliente Gemini está disponible antes de continuar
    if rating_system.gemini_client is None:
         print("\n🚫 No se puede proceder sin una conexión válida a la API de Gemini.")
         print("   Por favor, configura tu GEMINI_API_KEY correctamente.")
         return

    # Cargar archivos de mapas originales
    map_files = glob.glob(os.path.join(os.getcwd(), "mapa*.txt"))
    # Asegurarse de que solo carga los mapas con formato 'mapaXX.txt'
    map_files = sorted([f for f in map_files if os.path.basename(f).startswith("mapa") and os.path.basename(f)[4:-4].isdigit()])

    if not map_files:
        print("⚠️ No se encontraron archivos de mapas 'mapaXX.txt' válidos en el directorio actual.")
        # Opcional: Generar mapas artificiales aquí si no hay ninguno.
        # Por ahora, si no hay mapas, no podemos entrenar un transformer.
        return

    print(f"Encontrados {len(map_files)} archivos de mapas originales.")

    # Obtener mapas con ponderación basada en calidad (usando rating_system)
    print("Preparando mapas para entrenamiento con pesos...")
    weighted_files, weights = rating_system.get_weighted_maps(map_files)
    print("✓ Mapas originales ponderados según calidad y estructura.")

    # Crear tokenizador (basado en el vocabulario fijo)
    tokenizer = MapTokenizer()
    print(f"Vocabulario fijo: {tokenizer.vocab_size} tokens.")

    # Crear dataset ponderado para entrenamiento
    weighted_dataset = WeightedMapDataset(tokenizer, weighted_files, weights)
    
    # Verificar si hay ejemplos válidos en el dataset
    if not weighted_dataset or len(weighted_dataset) < BATCH_SIZE:
         print(f"⚠️ No hay suficientes mapas válidos ({len(weighted_dataset)} < {BATCH_SIZE}) para entrenar un batch.")
         print("   Asegúrate de que tus archivos 'mapaXX.txt' existen y contienen J, |>, y ■.")
         # Opcional: Generar mapas artificiales si no hay suficientes.
         # Por ahora, salimos.
         return


    # Crear un sampler ponderado
    weighted_sampler = weighted_dataset.get_sampler()

    # Crear dataloader con el sampler ponderado y collate_fn
    train_dataloader = DataLoader(
        weighted_dataset,
        batch_size=BATCH_SIZE,
        sampler=weighted_sampler,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_idx),
        num_workers=0 # num_workers > 0 puede causar problemas con WeightedRandomSampler y collate_fn custom
    )

    # Crear un dataset normal para validación (sin ponderación)
    # Usamos el mismo conjunto de archivos, pero un Dataset simple y split
    full_dataset = MapDataset(tokenizer, map_files)
    
    # Verificar si hay suficientes ejemplos para split de validación
    if len(full_dataset) < 10: # Necesitamos al menos ~10 mapas para un split razonable
         print(f"⚠️ No hay suficientes mapas originales ({len(full_dataset)}) para crear un conjunto de validación. Usando todos los datos para entrenamiento.")
         val_dataloader = None # No hay validación
    else:
         train_size = int(0.9 * len(full_dataset))
         val_size = len(full_dataset) - train_size
         _, val_dataset = random_split(full_dataset, [train_size, val_size])

         val_dataloader = DataLoader(
             val_dataset,
             batch_size=BATCH_SIZE,
             shuffle=False,
             collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_idx),
             num_workers=0
         )


    # Crear modelo Transformer
    model = MapTransformerModel(vocab_size=tokenizer.vocab_size)
    print(f"Creado modelo con {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"Verificando modelo: vocab_size={model.vocab_size}, output_layer.out_features={model.output_layer.out_features}")
    
    # **VALIDACIÓN CRÍTICA: Asegurar que el modelo tenga el tamaño correcto**
    if model.output_layer.out_features != tokenizer.vocab_size:
        print(f"❌ ERROR CRÍTICO: Tamaño de output_layer ({model.output_layer.out_features}) != vocab_size ({tokenizer.vocab_size})")
        print("Esto causará errores de CUDA. Verificar configuración del modelo.")
        return
    
    model.to(device) # Mover modelo al dispositivo

    # Entrenar modelo o cargar uno existente
    # Buscamos el modelo guardado general
    best_model_path = 'best_map_transformer.pt'
    if os.path.exists(best_model_path):
        print(f"Cargando modelo existente: {best_model_path}")
        try:
             checkpoint = torch.load(best_model_path, map_location=device)
             model.load_state_dict(checkpoint['model_state_dict'])
             print(f"Modelo cargado (epoch: {checkpoint.get('epoch', 'N/A')}, val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        except Exception as e:
             print(f"Error loading {best_model_path}: {e}. Starting fresh.")

        # Preguntar si quiere continuar entrenamiento
        train_more = input("¿Deseas continuar el entrenamiento? (s/n): ").lower() == 's'

        if train_more:
            print("Continuando entrenamiento con evaluación Gemini...")
            train_model(model, train_dataloader, val_dataloader, tokenizer, device,
                      epochs=EPOCHS, lr=LEARNING_RATE, rating_system=rating_system)
        else:
            # Si no entrena más, saltamos directamente a generar mapas finales
            pass # No necesitamos hacer nada aquí, la generación final está más abajo

    else:
        # Si no existe best_map_transformer.pt, entrenar desde cero
        print("No se encontró modelo 'best_map_transformer.pt'. Entrenando nuevo modelo...")
        train_model(model, train_dataloader, val_dataloader, tokenizer, device,
                   epochs=EPOCHS, lr=LEARNING_RATE, rating_system=rating_system)

    # --- Generar mapas finales ---
    # Cargamos el mejor modelo si se entrenó o si ya existía al inicio
    # El entrenamiento guarda 'best_map_transformer.pt' y potencialmente 'model_quality>XX.pt'
    quality_model_path = f'model_quality>{QUALITY_THRESHOLD}_*.pt'
    found_quality_models = glob.glob(quality_model_path)
    
    final_model_to_load = best_model_path # Por defecto carga el best_map_transformer

    if found_quality_models:
        # Cargar el modelo de calidad más reciente (basado en nombre de archivo)
        final_model_to_load = max(found_quality_models)
        print(f"\nEncontrado modelo que superó el umbral de calidad: {final_model_to_load}. Cargando este modelo para generación final.")
    elif os.path.exists('model_finished_e*.pt'):
         # Si no hubo early stop, cargar el modelo de la última época
         finished_models = glob.glob('model_finished_e*.pt')
         if finished_models:
              final_model_to_load = max(finished_models)
              print(f"\nCargando modelo del final del entrenamiento: {final_model_to_load} para generación final.")


    # Cargar el modelo seleccionado para la generación final
    if os.path.exists(final_model_to_load):
        try:
            checkpoint = torch.load(final_model_to_load, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo '{final_model_to_load}' cargado para generación final.")
        except Exception as e:
            print(f"❌ Error loading model for final generation {final_model_to_load}: {e}. Usando el modelo actual en memoria (si existe).")
    else:
         print("⚠️ No se encontró ningún modelo final guardado ('best_map_transformer.pt', 'model_quality>XX.pt', 'model_finished_e*.pt').")
         print("   Usando el modelo en memoria del inicio de la ejecución (no entrenado si era nuevo).")


    # Generar mapas finales con el modelo cargado/entrenado
    generate_maps_for_output(model, tokenizer, device, rating_system, num_maps=NUM_MAPS_GEN)

    print("\n✨ ¡Proceso completado! ✨")


if __name__ == "__main__":
    main()