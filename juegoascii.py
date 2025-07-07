import pygame
import sys
import time
import os
import random

# Inicializar pygame
pygame.init()

# Dimensiones iniciales del mapa y pantalla
MAP_WIDTH = 200
MAP_HEIGHT = 30
FONT_SIZE = 30
VIEW_WIDTH = 30
VIEW_HEIGHT = 18
SCREEN_WIDTH = VIEW_WIDTH * FONT_SIZE
SCREEN_HEIGHT = VIEW_HEIGHT * FONT_SIZE

# Configurar fuente
pygame.font.init()
font = pygame.font.SysFont('Courier New', FONT_SIZE)

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
STAR_COLOR = (200, 200, 200)


# Crear pantalla
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super ASCII Bro")

# Función para cargar el mapa desde un archivoS
def load_map_from_file(filename="mapa.txt"):
    with open(filename, 'r', encoding='utf-8') as file:
        map_data = file.read().splitlines()
    
    actual_height = len(map_data)
    actual_width = max(len(line) for line in map_data) if map_data else MAP_WIDTH
    
    map_width = max(actual_width, MAP_WIDTH)
    map_height = max(actual_height, MAP_HEIGHT)
        
    for i in range(len(map_data)):
        line_len = len(map_data[i])
        if line_len < map_width:
            map_data[i] = map_data[i] + " " * (map_width - line_len)
    
    if len(map_data) < map_height:
        map_data.extend([" " * map_width] * (map_height - len(map_data)))
            
    return map_data, map_width, map_height
    
def show_intro_animation(screen):
    # Configuración inicial
    title = "Super ASCII Bro"
    background_color = BLACK
    title_color = WHITE
    player_color = BLUE
    
    # Crear estrellas para el fondo
    stars = []
    for _ in range(50):
        x = random.randint(0, SCREEN_WIDTH // FONT_SIZE - 1)
        y = random.randint(0, SCREEN_HEIGHT // FONT_SIZE - 1)
        stars.append((x, y))
    
    # Estados de la animación
    WALKING_TO_JUMP = 0   # Estado inicial, camina hasta el punto de salto
    JUMPING = 1           # Saltando
    FALLING = 2           # Cayendo
    WALKING_AFTER_JUMP = 3  # Nuevo estado: caminar después del salto
    animation_state = WALKING_TO_JUMP
    
    # Sprite del jugador usando las mismas animaciones que en el juego
    standing = [" O ", "/|\\", "/ \\"]
    walking1 = [" O ", "/|\\", "| \\"]
    walking2 = [" O ", "/|\\", "/ |"]
    jumping_up = [" O ", "/|\\", "/  \\"]
    jumping_peak = [" O ", "-|-", "/ \\"]
    jumping_down = [" O ", "\\|/", "/ \\"]
    
    player_x = -5  # Comienza fuera de la pantalla a la izquierda
    player_y = (SCREEN_HEIGHT // FONT_SIZE) - 4  # Parte inferior (dejando espacio para saltar)
    walk_frame = 0  # Frame actual de la animación del jugador
    jump_power = 0  # Para controlar la potencia del salto
    
    # Animación del título
    title_letters = list(title)
    displayed_title = ""
    title_index = 0
    title_complete = False
    
    # Control de tiempo
    clock = pygame.time.Clock()
    
    # Bucle de la animación - ahora continúa hasta que el jugador sale de pantalla
    animation_active = True
    while animation_active:
        screen.fill(background_color)
        
        # Dibujar estrellas con parpadeo
        for star in stars:
            if random.random() > 0.1:  # 90% de probabilidad de visibilidad
                star_surface = font.render("*", True, STAR_COLOR)
                screen.blit(star_surface, (star[0] * FONT_SIZE, star[1] * FONT_SIZE))
        
        # Sección de animación de escritura del título
        if not title_complete:
            if title_index < len(title_letters):
                displayed_title += title_letters[title_index]
                title_index += 1
                pygame.time.wait(100)  # Retraso entre letras
                if title_index == len(title_letters):
                    title_complete = True
        
        # Siempre mostrar el título, independientemente de si la animación de escritura terminó
        title_surface = font.render(displayed_title, True, title_color)
        screen.blit(title_surface, (SCREEN_WIDTH // 2 - title_surface.get_width() // 2, 100))
        
        # Actualizar la animación del jugador
        current_sprite = None
        
        # ESTADOS DE ANIMACIÓN
        if animation_state == WALKING_TO_JUMP:
            # Caminar hasta llegar a un tercio de la pantalla
            if player_x < SCREEN_WIDTH // (FONT_SIZE * 3):
                player_x += 0.5
                if int(player_x) % 5 == 0:  # Cambiar frame cada 5 pasos
                    walk_frame = (walk_frame + 1) % 2
                
                # Seleccionar sprite para caminar
                if walk_frame == 0:
                    current_sprite = walking1
                else:
                    current_sprite = walking2
            else:
                # Comenzar a saltar en un tercio de la pantalla
                animation_state = JUMPING
                jump_power = 6  # Poder de salto
        
        elif animation_state == JUMPING:
            # Movimiento de salto - más suave
            if jump_power > 0:
                player_y -= 0.5  # Movimiento más suave, subir medio bloque por frame
                jump_power -= 1
                player_x += 0.3  # Avance horizontal durante el salto
                
                if jump_power > 3:
                    current_sprite = jumping_up
                else:
                    current_sprite = jumping_peak
            else:
                animation_state = FALLING
        
        elif animation_state == FALLING:
            # Caer después de saltar - más suave
            if player_y < (SCREEN_HEIGHT // FONT_SIZE) - 4:
                player_y += 0.5  # Movimiento más suave, caer medio bloque por frame
                player_x += 0.3  # Seguir avanzando horizontalmente
                current_sprite = jumping_down
            else:
                # Cambiar al nuevo estado después de saltar
                animation_state = WALKING_AFTER_JUMP
                
        elif animation_state == WALKING_AFTER_JUMP:
            # Seguir caminando hasta salir de pantalla sin volver a saltar
            player_x += 0.5
            if int(player_x) % 5 == 0:
                walk_frame = (walk_frame + 1) % 2
            
            if walk_frame == 0:
                current_sprite = walking1
            else:
                current_sprite = walking2
                
        # Si no se ha definido un sprite, usar el sprite de pie
        if not current_sprite:
            current_sprite = standing
            
        # Dibujar al jugador
        if player_x >= 0:  # Solo dibujar cuando está visible
            for i, line in enumerate(current_sprite):
                player_surface = font.render(line, True, player_color)
                screen.blit(player_surface, (int(player_x) * FONT_SIZE, (player_y + i) * FONT_SIZE))
        
        # Terminar la animación si el personaje sale de la pantalla
        if player_x > SCREEN_WIDTH // FONT_SIZE:
            animation_active = False
        
        pygame.display.flip()
        clock.tick(25)  # FPS ajustados para una animación más fluida
        
        # Permitir salir con ESC o haciendo clic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                animation_active = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                animation_active = False
    
    # Transición rápida (fade-out)
    for alpha in range(255, 0, -15):  # Fade-out más rápido
        temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        temp_surface.blit(screen, (0, 0))
        temp_surface.set_alpha(alpha)
        screen.fill(background_color)
        screen.blit(temp_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

# Clase Player (sin cambios en funcionalidad)
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 3
        self.height = 3
        self.is_jumping = False
        self.jump_power = 0
        self.jump_stage = 0
        self.is_crouching = False
        self.walk_index = 0
        self.direction = 1
        self.last_ground_time = 0
        self.is_moving = False
        
        self.standing = [" O ", "/|\\", "/ \\"]
        self.walking1 = [" O ", "/|\\", "| \\"]
        self.walking2 = [" O ", "/|\\", "/ |"]
        self.jumping_up = [" O ", "/|\\", "/  \\"]
        self.jumping_peak = [" O ", "-|-", "/ \\"]
        self.jumping_down = [" O ", "\\|/", "/ \\"]
        self.crouching = [" O ", "/_\\"]
        self.crouching_walk1 = [" O ", "/\\_"]  # Ambas piernas se mueven juntas hacia la derecha
        self.crouching_walk2 = [" O ", "_/\\"]
    
    def get_current_sprite(self):
        if self.is_crouching:
            if self.is_moving:
                if self.walk_index == 0:
                    return self.crouching
                elif self.walk_index == 1:
                    return self.crouching_walk1
                else:
                    return self.crouching_walk2
            else:
                return self.crouching
        
        if self.is_jumping:
            if self.jump_stage == 0:
                return self.jumping_up
            elif self.jump_stage == 1:
                return self.jumping_peak
            else:
                return self.jumping_down
        else:
            if self.walk_index == 0:
                return self.standing
            elif self.walk_index == 1:
                return self.walking1
            else:
                return self.walking2
    
    def get_rect(self):
        height = 2 if self.is_crouching else 3
        return (self.x, self.y, self.x + 2, self.y + height - 1)

# Clase Enemy (sin cambios en funcionalidad)
class Enemy:
    def __init__(self, x, y, left_limit, right_limit):
        self.x = x
        self.y = y
        self.left_limit = left_limit
        self.right_limit = right_limit
        self.direction = 1
        self.alive = True
        self.sprite = "☉"
    
    def update(self):
        if self.alive:
            new_x = self.x + self.direction
            if new_x <= self.left_limit or new_x >= self.right_limit:
                self.direction = -self.direction
            else:
                self.x = new_x
    
    def get_rect(self):
        return (self.x, self.y, self.x, self.y)

# Clase Game (sin cambios en funcionalidad principal)
class Game:
    def __init__(self, map_filename="mapa.txt"):
        self.game_map, map_width, map_height = load_map_from_file(map_filename)
        self.original_map = [row[:] for row in self.game_map]

        global MAP_WIDTH, MAP_HEIGHT
        MAP_WIDTH = map_width
        MAP_HEIGHT = map_height

        self.disappearing_tiles = set()
        self.disappear_timer = {}
        self.score = 0
        self.camera_x = 0
        self.camera_y = 0
        self.game_over = False
        self.victory = False
        self.player = Player(3, 3)
        self.enemies = []
        self.initialize_game_objects()
        
    def initialize_game_objects(self):
        for y in range(len(self.game_map)):
            for x in range(len(self.game_map[y])):
                if self.game_map[y][x] == "J":
                    self.player.x = x
                    for ground_y in range(y + 1, MAP_HEIGHT):
                        if "■" in self.game_map[ground_y]:
                            self.player.y = ground_y - self.player.height
                            break
                    else:
                        self.player.y = y
                    self.game_map[y] = self.game_map[y][:x] + " " + self.game_map[y][x+1:]
                    self.original_map[y] = self.original_map[y][:x] + " " + self.original_map[y][x+1:]
                elif self.game_map[y][x] == "E":
                    enemy_x = x
                    enemy_y = y
                    left_limit = max(0, enemy_x - 5)
                    right_limit = min(MAP_WIDTH - 1, enemy_x + 5)
                    enemy = Enemy(enemy_x, enemy_y, left_limit, right_limit)
                    self.enemies.append(enemy)
                    self.game_map[y] = self.game_map[y][:x] + " " + self.game_map[y][x+1:]
                    self.original_map[y] = self.original_map[y][:x] + " " + self.original_map[y][x+1:]
                elif self.game_map[y][x] == "☉":
                    enemy_x = x
                    enemy_y = y
                    left_x = x - 1
                    while left_x >= 0 and self.game_map[y][left_x] != "■":
                        left_x -= 1
                    right_x = x + 1
                    while right_x < MAP_WIDTH and self.game_map[y][right_x] != "■":
                        right_x += 1
                    left_limit = left_x + 1
                    right_limit = right_x - 1
                    enemy = Enemy(enemy_x, enemy_y, left_limit, right_limit)
                    self.enemies.append(enemy)
                    self.game_map[y] = self.game_map[y][:x] + " " + self.game_map[y][x+1:]
                    self.original_map[y] = self.original_map[y][:x] + " " + self.original_map[y][x+1:]
        
        if self.player.y == 3:
            for y in range(MAP_HEIGHT - 1, -1, -1):
                if "■" in self.game_map[y]:
                    self.player.y = y - self.player.height
                    break
    
    def try_stand_up(self):
        if self.can_move(self.player.x, self.player.y - 1):
            self.player.y -= 1
            self.player.is_crouching = False
            return True
        return False

    def can_move(self, new_x, new_y):
        if new_x < 0 or new_x >= MAP_WIDTH - 2:
            return False
        height = 2 if self.player.is_crouching else 3
        for i in range(3):
            for j in range(height):
                check_y = new_y + j
                check_x = new_x + i
                if 0 <= check_y < MAP_HEIGHT and 0 <= check_x < MAP_WIDTH:
                    if self.game_map[check_y][check_x] in ["■", "□"]:
                        return False
        return True
    
    def is_on_ground(self):
        height = 2 if self.player.is_crouching else 3
        feet_y = self.player.y + (height - 1)
        if feet_y + 1 >= MAP_HEIGHT:
            return True
        for i in range(3):
            check_x = self.player.x + i
            if 0 <= check_x < MAP_WIDTH and feet_y + 1 < MAP_HEIGHT:
                if self.game_map[feet_y + 1][check_x] in ["■", "□"]:
                    return True
        return False
    
    def update_disappearing_tiles(self):
        current_time = time.time()
        tiles_to_restore = []
        for pos, restore_time in list(self.disappear_timer.items()):
            if current_time >= restore_time:
                x, y = pos
                if self.original_map[y][x] == "■":
                    self.game_map[y] = self.game_map[y][:x] + "■" + self.game_map[y][x+1:]
                tiles_to_restore.append(pos)
        for pos in tiles_to_restore:
            self.disappearing_tiles.remove(pos)
            del self.disappear_timer[pos]
    
    def collect_item(self, x, y):
        if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH:
            if self.game_map[y][x] == "C":
                self.score += 5
                self.game_map[y] = self.game_map[y][:x] + " " + self.game_map[y][x+1:]
                return True
        return False
    
    def check_collision_with_enemies(self):
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            player_rect = self.player.get_rect()
            enemy_rect = enemy.get_rect()
            if (self.player.jump_stage == 2 and 
                player_rect[0] <= enemy_rect[0] <= player_rect[2] and 
                player_rect[3] >= enemy_rect[1] - 1 and player_rect[3] <= enemy_rect[1] + 1):
                enemy.alive = False
                self.score += 10
                self.player.is_jumping = True
                self.player.jump_power = 2
                self.player.jump_stage = 0
                return False
            if (player_rect[0] <= enemy_rect[0] <= player_rect[2] or player_rect[0] <= enemy_rect[2] <= player_rect[2]) and \
               (player_rect[1] <= enemy_rect[1] <= player_rect[3] or player_rect[1] <= enemy_rect[3] <= player_rect[3]):
                self.game_over = True
                return True
        return False
    
    def check_victory(self):
        for i in range(3):
            for j in range(2 if self.player.is_crouching else 3):
                if (self.player.y + j < MAP_HEIGHT and self.player.x + i < MAP_WIDTH and 
                    self.game_map[self.player.y + j][self.player.x + i] == "|"):
                    self.victory = True
                    return True
        return False
    
    def update(self):
        if self.game_over or self.victory:
            return
        self.update_disappearing_tiles()
        current_time = time.time()
        on_ground = self.is_on_ground()
        if on_ground:
            self.player.last_ground_time = current_time
        if self.player.is_jumping:
            if self.player.jump_power > 0:
                block_broken = False
                if self.player.jump_stage == 0:
                    player_center_x = self.player.x + 1
                    breakable_blocks = []
                    for i in range(3):
                        check_x = self.player.x + i
                        check_y = self.player.y - 1
                        if (0 <= check_y < MAP_HEIGHT and 0 <= check_x < MAP_WIDTH and 
                            self.game_map[check_y][check_x] == "□"):
                            distance = abs(check_x - player_center_x)
                            breakable_blocks.append((check_x, check_y, distance))
                    if breakable_blocks:
                        closest_block = min(breakable_blocks, key=lambda block: block[2])
                        x, y, _ = closest_block
                        self.game_map[y] = self.game_map[y][:x] + " " + self.game_map[y][x+1:]
                        self.score += 2
                        block_broken = True
                if block_broken:
                    self.player.jump_power = 0
                    self.player.jump_stage = 2
                elif self.can_move(self.player.x, self.player.y - 1):
                    self.player.y -= 1
                    for i in range(3):
                        for j in range(2 if self.player.is_crouching else 3):
                            self.collect_item(self.player.x + i, self.player.y + j)
                self.player.jump_power -= 1
                if self.player.jump_power == 2:
                    self.player.jump_stage = 1
                elif self.player.jump_power <= 0:
                    self.player.jump_stage = 2
            else:
                if on_ground:
                    self.player.is_jumping = False
                    self.player.jump_stage = 0
                else:
                    self.player.jump_stage = 2
        if not on_ground and (not self.player.is_jumping or self.player.jump_power <= 0):
            if self.can_move(self.player.x, self.player.y + 1):
                self.player.y += 1
                for i in range(3):
                    for j in range(2 if self.player.is_crouching else 3):
                        self.collect_item(self.player.x + i, self.player.y + j)
            if self.player.jump_stage != 2:
                self.player.jump_stage = 2
            if self.player.y > MAP_HEIGHT - 4:
                self.game_over = True
        for enemy in self.enemies:
            if enemy.alive:
                new_x = enemy.x + enemy.direction
                if new_x <= enemy.left_limit or new_x >= enemy.right_limit:
                    enemy.direction = -enemy.direction
                else:
                    enemy.x = new_x
        self.check_collision_with_enemies()
        self.check_victory()
        deadzone_half_width = 1
        deadzone_half_height = 2
        player_center_x = self.player.x + 1
        player_center_y = self.player.y + (1 if self.player.is_crouching else 1.5)
        deadzone_left = self.camera_x + VIEW_WIDTH // 2 - deadzone_half_width
        deadzone_right = self.camera_x + VIEW_WIDTH // 2 + deadzone_half_width
        deadzone_top = self.camera_y + VIEW_HEIGHT // 2 - deadzone_half_height
        deadzone_bottom = self.camera_y + VIEW_HEIGHT // 2 + deadzone_half_height
        target_camera_x = self.camera_x
        if player_center_x < deadzone_left:
            target_camera_x = self.camera_x - (deadzone_left - player_center_x)
        elif player_center_x > deadzone_right:
            target_camera_x = self.camera_x + (player_center_x - deadzone_right)
        target_camera_y = self.camera_y
        if player_center_y < deadzone_top:
            target_camera_y = self.camera_y - (deadzone_top - player_center_y)
        elif player_center_y > deadzone_bottom:
            target_camera_y = self.camera_y + (player_center_y - deadzone_bottom)
        target_camera_x = max(0, min(target_camera_x, MAP_WIDTH - VIEW_WIDTH))
        vertical_offset = 2
        if target_camera_y < 0 and target_camera_y > -vertical_offset:
            pass
        elif target_camera_y < -vertical_offset:
            target_camera_y = -vertical_offset
        elif target_camera_y > MAP_HEIGHT - VIEW_HEIGHT:
            target_camera_y = MAP_HEIGHT - VIEW_HEIGHT
        self.camera_x = int(target_camera_x)
        self.camera_y = int(target_camera_y)

    def render(self, screen):
        screen.fill(BLACK)
        cam_y = int(self.camera_y)
        cam_x = int(self.camera_x)
        for y in range(cam_y, min(cam_y + VIEW_HEIGHT, MAP_HEIGHT)):
            for x in range(cam_x, min(cam_x + VIEW_WIDTH, MAP_WIDTH)):
                screen_x = (x - cam_x) * FONT_SIZE
                screen_y = (y - cam_y) * FONT_SIZE
                if 0 <= y < len(self.game_map) and 0 <= x < len(self.game_map[y]):
                    char = self.game_map[y][x]
                    if char == "■":
                        pygame.draw.rect(screen, WHITE, (screen_x, screen_y, FONT_SIZE, FONT_SIZE))
                        continue
                    if char == "º":
                        continue
                    color = WHITE
                    if char == "C":
                        color = YELLOW
                    elif char == "|":
                        color = RED
                    elif char == "🟦":
                        color = LIGHT_BLUE
                    elif char == "□":
                        color = (200, 140, 80)
                    if char != " ":
                        text_surface = font.render(char, True, color)
                        screen.blit(text_surface, (screen_x, screen_y))
        for enemy in self.enemies:
            if enemy.alive:
                screen_x = (enemy.x - self.camera_x) * FONT_SIZE
                screen_y = (enemy.y - self.camera_y) * FONT_SIZE
                if 0 <= screen_x < SCREEN_WIDTH and 0 <= screen_y < SCREEN_HEIGHT:
                    text_surface = font.render(enemy.sprite, True, RED)
                    screen.blit(text_surface, (screen_x, screen_y))
        sprite = self.player.get_current_sprite()
        height = 2 if self.player.is_crouching else 3
        for i in range(height):
            screen_x = (self.player.x - self.camera_x) * FONT_SIZE
            screen_y = (self.player.y + i - self.camera_y) * FONT_SIZE
            if i < len(sprite) and 0 <= screen_y < SCREEN_HEIGHT:
                text_surface = font.render(sprite[i], True, BLUE)
                screen.blit(text_surface, (screen_x, screen_y))
        score_text = f"Puntuación: {self.score}"
        text_surface = font.render(score_text, True, WHITE)
        screen.blit(text_surface, (10, 10))
        if self.game_over:
            game_over_text = "¡Perdiste! 😢 Presiona R para reiniciar"
            text_surface = font.render(game_over_text, True, RED)
            screen.blit(text_surface, (
                SCREEN_WIDTH // 2 - text_surface.get_width() // 2, 
                SCREEN_HEIGHT // 2 - text_surface.get_height() // 2
            ))
        elif self.victory:
            victory_text = "¡Ganaste! 🎉 Presiona R para reiniciar"
            text_surface = font.render(victory_text, True, GREEN)
            screen.blit(text_surface, (
                SCREEN_WIDTH // 2 - text_surface.get_width() // 2, 
                SCREEN_HEIGHT // 2 - text_surface.get_height() // 2
            ))

    def reset(self):
        self.__init__()

# Menú de selección de nivel (mejorado, sin afectar funcionalidad)
def select_map_menu(screen):
    maps = [f for f in os.listdir('.') if f.startswith('mapa') and f.endswith('.txt') and f[4:-4].isdigit()]
    if not maps:
        maps = ["mapa.txt"]
    
    selected = 0
    offset = 0
    max_visible = 6  # Menos ítems para dar más espacio
    fh = FONT_SIZE
    margin_top = 50
    margin_between = 10

    while True:
        screen.fill(BLACK)

        # 1) Título
        title_surf = font.render("Selecciona un nivel", True, WHITE)
        screen.blit(title_surf, ((SCREEN_WIDTH - title_surf.get_width()) // 2, margin_top))

        # 2) Flecha arriba
        if offset > 0:
            up_surf = font.render("⬆ Niveles anteriores", True, GREEN)
            screen.blit(up_surf, ((SCREEN_WIDTH - up_surf.get_width()) // 2, margin_top + fh + margin_between))

        # 3) Lista de mapas
        start_y = margin_top + 2*(fh + margin_between)
        for idx in range(offset, min(offset + max_visible, len(maps))):
            y = start_y + (idx - offset) * (fh + margin_between)
            color = YELLOW if idx == selected else WHITE
            item_surf = font.render(f"{idx+1}. {maps[idx]}", True, color)
            screen.blit(item_surf, ((SCREEN_WIDTH - item_surf.get_width()) // 2, y))

        # 4) Flecha abajo
        end = offset + max_visible
        if end < len(maps):
            down_surf = font.render("⬇ Niveles siguientes", True, GREEN)
            y_down = start_y + max_visible * (fh + margin_between) + margin_between
            screen.blit(down_surf, ((SCREEN_WIDTH - down_surf.get_width()) // 2, y_down))

        # 5) Indicador de página
        if len(maps) > max_visible:
            page = offset // max_visible + 1
            total = (len(maps)-1)//max_visible + 1
            pg_surf = font.render(f"Página {page}/{total}", True, WHITE)
            screen.blit(pg_surf, (SCREEN_WIDTH - pg_surf.get_width() - 20, SCREEN_HEIGHT - fh*3-20))

        # 6) Barra de instrucciones (2 líneas)
        instr1 = "W/S ↑↓ Seleccionar  |  Enter Confirmar"
        instr2 = "A/D ←→ Página  |  Esc Salir"
        ins1_s = font.render(instr1, True, LIGHT_BLUE)
        ins2_s = font.render(instr2, True, LIGHT_BLUE)
        y_base = SCREEN_HEIGHT - fh*2 - margin_between
        screen.blit(ins1_s, ((SCREEN_WIDTH - ins1_s.get_width()) // 2, y_base))
        screen.blit(ins2_s, ((SCREEN_WIDTH - ins2_s.get_width()) // 2, y_base + fh + margin_between//2))

        pygame.display.flip()

        # --- Manejo de eventos igual que antes ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = min(selected + 1, len(maps) - 1)
                    if selected >= offset + max_visible:
                        offset += 1
                elif event.key in (pygame.K_UP, pygame.K_w):
                    selected = max(selected - 1, 0)
                    if selected < offset:
                        offset -= 1
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    # Página siguiente
                    offset = min(offset + max_visible, len(maps) - max_visible)
                    selected = offset
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    # Página anterior
                    offset = max(offset - max_visible, 0)
                    selected = offset
                elif event.key in (pygame.K_ESCAPE,):
                    return None  # O maneja como prefieras
                elif event.key == pygame.K_RETURN:
                    return maps[selected]




# Menú de pausa (mejorado y corregido)
def show_pause_menu(screen):
    lines = ["Pausa", "R - Reanudar", "M - Menú Principal", "Q - Salir"]
    while True:
        screen.fill(BLACK)
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (SCREEN_WIDTH // 2 - text_surface.get_width() // 2, SCREEN_HEIGHT // 2 - 100 + i * 40))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return 'resume'
                elif event.key == pygame.K_m:
                    return 'menu'
                elif event.key == pygame.K_q:
                    return 'quit'

# Función principal (ajustada para respetar funcionalidad)
def main():
    global screen

    show_intro_animation(screen)

    while True:
        map_filename = select_map_menu(screen)
        game = Game(map_filename)
        clock = pygame.time.Clock()
        frame_count = 0
        running = True
        while running:
            frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and (game.game_over or game.victory):
                        game = Game(map_filename)
                    elif event.key == pygame.K_ESCAPE:
                        pause_choice = show_pause_menu(screen)
                        if pause_choice == 'resume':
                            continue
                        elif pause_choice == 'menu':
                            running = False  # Volver al menú principal
                            break
                        elif pause_choice == 'quit':
                            pygame.quit()
                            sys.exit()
            if not game.game_over and not game.victory:
                keys = pygame.key.get_pressed()
                current_time = time.time()
                coyote_time_active = current_time - game.player.last_ground_time <= 0.2
                if frame_count % 3 == 0:
                    game.player.is_moving = False
                    if keys[pygame.K_d] and game.can_move(game.player.x + 1, game.player.y):
                        game.player.x += 1
                        game.player.walk_index = (game.player.walk_index + 1) % 3
                        game.player.direction = 1
                        game.player.is_moving = True
                        for i in range(3):
                            for j in range(2 if game.player.is_crouching else 3):
                                game.collect_item(game.player.x + i, game.player.y + j)
                    elif keys[pygame.K_a] and game.can_move(game.player.x - 1, game.player.y):
                        game.player.x -= 1
                        game.player.walk_index = (game.player.walk_index + 1) % 3
                        game.player.direction = -1
                        game.player.is_moving = True
                        for i in range(3):
                            for j in range(2 if game.player.is_crouching else 3):
                                game.collect_item(game.player.x + i, game.player.y + j)
                    if keys[pygame.K_w]:
                        if game.player.is_crouching:
                            game.try_stand_up()
                        elif not game.player.is_jumping and (game.is_on_ground() or coyote_time_active):
                            game.player.is_jumping = True
                            game.player.jump_power = 6
                            game.player.jump_stage = 0
                    if keys[pygame.K_s] and not game.player.is_jumping:
                        game.player.is_crouching = True
                    elif not keys[pygame.K_s] and game.player.is_crouching:
                        game.try_stand_up()
                    game.update()
            game.render(screen)
            pygame.display.flip()
            clock.tick(30)
        if not running and pause_choice != 'menu':
            break
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()