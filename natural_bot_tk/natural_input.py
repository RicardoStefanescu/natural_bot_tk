import string
import time

import numpy as np
import pyautogui
pyautogui.MINIMUM_DURATION = 0  # Default: 0.1
pyautogui.MINIMUM_SLEEP = 0  # Default: 0.05
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # Set to false in real world

from .resources.keyboard_layouts import us_layout


class Mouse:
    def __init__(self, screen_size):
        self._s_width, self._s_height = screen_size

    def left_click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.click(pyautogui.RIGHT)

    def move_to(self, p_destination, 
                max_deviation=0.05, 
                n_steps=None, 
                linear_progression=False,
                noisiness=0.4, 
                max_noise_deviation=0.9):
        '''
        Mueve el raton a un pixel dadas sus coordenadas.\n
        `max_deviation`: La desviacion maxima de los puntos de control\n
        `n_steps`: El numero de puntos de la curva que calcular\n
        `linear_progression`: \n
          - True: usamos una distribucion lineal para los puntos\n
          - False: Usamos una distribucion triangular para los puntos\n
        `noisiness` - Valor [0-1) que determina a que porcentaje de puntos anadir ruido \n
        `max_noise_deviation` - Desviacion maxima 
        '''
        t_d = 0.001
        p_origin = pyautogui.position()

        # Normalize values to [0:1)
        x_o = p_origin[0]/self._s_width
        y_o = p_origin[1]/self._s_height
        x_d = p_destination[0]/self._s_width
        y_d = p_destination[1]/self._s_height

        # Generate Bezier curve
        p_0 = np.array([x_o, y_o])
        p_3 = np.array([x_d, y_d])
        points = self.generate_cubic_bezier(p_0, p_3, max_deviation=max_deviation, n_steps=n_steps, linear_progression=linear_progression)

        # Bring points back to pixels
        points[:, 0] *= self._s_width
        points[:, 1] *= self._s_height

        # Add noise
        points = self.add_noise(points, noisiness, max_noise_deviation)

        # Move
        for p in points:
            time.sleep(t_d)
            pyautogui.moveTo(p[0], p[1])

    def chain_clicks(self, destinations, randomize=False):
        if randomize:
            np.random.shuffle(destinations)

        for d in destinations:
            self.move_to(d, max_deviation=0.01)
            self.left_click()

    def generate_cubic_bezier(self, p_0, p_3, 
                                max_deviation=0.05,
                                n_steps=None, 
                                linear_progression=False):
        ''' Calcula los puntos de una curva de bezier.\n
        Los parametros son:\n
        `p_0`: El punto inicial e.g. [0,0]\n
        `p_3`: El punto final e.g. [10,10]\n
        `max_deviation`: La desviacion maxima de los puntos de control\n
        `n_steps`: El numero de puntos de la curva que calcular\n
        `linear_progression`: \n
          - True: usamos una distribucion lineal para los puntos\n
          - False: Usamos una distribucion triangular para los puntos
        '''
        # Calculamos los puntos de anclaje de la curva con la formula
        rand = max_deviation - np.random.uniform() * max_deviation * 2
        rand2 = max_deviation - np.random.uniform() * max_deviation * 2
        p_1 = p_0 + np.array([rand, rand2])

        rand = max_deviation - np.random.uniform() * max_deviation * 2
        rand2 = max_deviation - np.random.uniform() * max_deviation * 2
        p_2 = p_3 + np.array([rand, rand2])

        points = []

        # Calculamos los tiempos en los que movernos
        # el numero de puntos intermedios sera proporcional a la distancia
        steps = np.random.randint(800, 1000) if n_steps is None else n_steps
        if linear_progression:
            T = np.array(range(0, steps+1)) / steps
        else:
            T = np.sort(np.random.triangular(0.0, np.random.uniform(0.9,1.), 1.0, steps))
            T = np.insert(T, 0, 0, axis=0)
            T = np.insert(T, len(T), 1, axis=0)

        for t_i in T:
            t = t_i
            s_0 = p_0*(1-t)**3
            s_1 = 3 * p_1 * t * (1 - t)**2
            s_2 = 3 * p_2 * (1-t) * t**2
            s_3 = p_3 * t**3

            p = s_0 + s_1 + s_2 + s_3

            points.append(p)

        return np.array(points)

    def add_noise(self, points, noisiness, max_deviation):
        '''
        `points` - Puntos a los que anadir ruido \n
        `noisiness` - Valor [0-1) que determina a que porcentaje de puntos anadir ruido \n
        `max_deviation` - Desviacion maxima 
        '''
        assert 1 > noisiness and noisiness >= 0
        assert len(points) != 0

        if noisiness == 0:
            return points

        # Calculate N points that we will create
        n_noisy_points = int(len(points) * noisiness)
        
        # Make it a pair number
        n_noisy_points = n_noisy_points if n_noisy_points % 2 == 0 else n_noisy_points - 1

        # Get a random set of points that we will add noise to
        i_noisy_points = []
        for _ in range(n_noisy_points):
            i = np.random.randint(1, len(points))

            # If we already took that index repeat
            while i in i_noisy_points:
                i = np.random.randint(1, len(points))

            i_noisy_points.append(i)

        # Calculate the vectors leading to those points
        vectors = []
        for i in i_noisy_points:
            vectors.append(points[i] - points[i - 1])
        vectors = np.array(vectors)

        # Choose pairs of vectors
        vs_1, vs_2 = np.split(vectors, 2)
        is_1, is_2 = np.split(np.array(i_noisy_points), 2)
        noise = np.zeros((len(points), 2))
        for i_1, i_2, v_1, v_2 in zip(is_1, is_2, vs_1, vs_2):
            # Calculate max deviations
            max_x_deviation = abs(min(v_1[0], v_2[0]))
            max_y_deviation = abs(min(v_1[1], v_2[1]))

            max_x_deviation *= max_deviation
            max_y_deviation *= max_deviation

            # Calculate noise
            n = np.empty(2)
            n[0] = np.random.uniform(-max_x_deviation, max_x_deviation)
            n[1] = np.random.uniform(-max_y_deviation, max_y_deviation)

            # Add opposite noise to each
            noise[i_1] += n
            noise[i_2] -= n

        new_points = []
        noise_sum = np.zeros(2)
        # delete the first point 
        new_points.append(points[0])

        # Recalculate points
        for og_p, n in zip(points[1:], noise[1:]):
            noise_sum += n
            new_p = og_p + noise_sum
            new_points.append(new_p)

        return np.array(new_points)

class Keyboard:
    def __init__(self, person_seed, layout=None):
        self._layout = us_layout if layout is None else layout
        self._key_to_index = {i:k for k,i in enumerate(self._layout.keys())}
        self._index_to_key = {k:i for k,i in enumerate(self._layout.keys())}

        # Unrandomized section
        np.random.seed(person_seed)

        self._handicap = np.random.uniform()
        self._move_delay_matrix = self._generate_move_delay_matrix()
        self._hold_delay_dict = self._generate_hold_delay_dict()

        np.random.seed()

    def type_text(self, text, stress=0.3):
        ''' Genera las teclas para el texto y las teclea'''

        keys = self.generate_keys(text, stress)

        # Press keys in that order
        try:
            for i, k in enumerate(keys):
                intended_key = k[0]
                needs_shift = k[1]
                movement_delay = k[2]
                hold_delay = k[3]

                time.sleep(movement_delay)
                if needs_shift:
                    self.hold_key('shift')

                self.press_key(intended_key, hold_delay)

                if i != len(keys)-1 and not keys[i+1][1]:
                    self.release_key('shift')
        except Exception as e:
            print(e)

    def generate_keys(self, text, stress):
        ''' Genera la secuencia de teclas y sus tiempos 
         para teclear un texto. \n
        `text`: El texto a teclear\n
        `stress`: Valor que determina la velocidad y la cantidad de errores.
        '''
        shifted_keys = '~!@#$%^&*()_+{}|:"<>?' + string.ascii_uppercase

        typo_posibility = stress * 0.2
        
        keys = []
        for i, c in enumerate(text):
            last_key = self.char_to_key(text[i - 1])
            this_key = self.char_to_key(c)

            if i != 0 and np.random.uniform() <= typo_posibility and this_key not in ['tab', 'return']:
                # Add typo and keys
                max_time = int(5 * stress)
                time_to_notice = 0 if max_time <= 1 else np.random.randint(1, max_time)
                time_to_notice = time_to_notice if time_to_notice < len(text) - i else len(text) - i
                key = self._generate_typo(c)

                origin_i = self._key_to_index[last_key]
                dest_i = self._key_to_index[key]

                movement_delay = self._move_delay_matrix[origin_i][dest_i] * 0.1 * self._handicap

                hold_delay = self._hold_delay_dict[key]
                needs_shift = key in shifted_keys

                # A key action has: key, wheter it needs shift, movement shift and hold delay    
                keys.append( [key, needs_shift, movement_delay, hold_delay] )

                _c = self.char_to_key(text[i])
                for j in range(time_to_notice):
                    _last_c = self.char_to_key(text[i + j - 1])
                    _c = self.char_to_key(text[i + j])
                    origin_i = self._key_to_index[_last_c]
                    dest_i = self._key_to_index[_c]

                    movement_delay = self._move_delay_matrix[origin_i][dest_i] * 0.1 * self._handicap

                    hold_delay = self._hold_delay_dict[_c]
                    needs_shift = _c in shifted_keys

                    # A key action has: key, wheter it needs shift, movement shift and hold delay    
                    keys.append( [_c, needs_shift, movement_delay, hold_delay] )

                # Backspace with delay
                origin_i = self._key_to_index[_c]
                dest_i = self._key_to_index['backspace']

                movement_delay = self._move_delay_matrix[origin_i][dest_i] * 0.1 * self._handicap
                hold_delay = self._hold_delay_dict['backspace']
                keys.append( ['backspace', False, movement_delay, hold_delay] )
                # Rest of backspaces
                for _ in range(time_to_notice):
                    keys.append( ['backspace', False, 0.05 + np.random.uniform(-0.005, 0.005), hold_delay] )

            if i == 0:
                movement_delay = 0
            else:
                origin_i = self._key_to_index[last_key]
                dest_i = self._key_to_index[this_key]

                movement_delay = self._move_delay_matrix[origin_i][dest_i] * 0.1 * self._handicap

            hold_delay = self._hold_delay_dict[this_key]
            needs_shift = c in shifted_keys

            # A key action has: key, wheter it needs shift, movement shift and hold delay    
            keys.append( [this_key, needs_shift, movement_delay, hold_delay] )

        return keys

    def press_key(self, char, delay=0.0):
        if char not in pyautogui.KEYBOARD_KEYS:
            raise Exception(f"{char} is not a valid key to press")

        pyautogui.keyDown(char)
        time.sleep(delay)
        pyautogui.keyUp(char)

    def hold_key(self, char):
        if char not in pyautogui.KEYBOARD_KEYS:
            raise Exception(f"{char} is not a valid key to press")

        pyautogui.keyDown(char)

    def release_key(self, char):
        if char not in pyautogui.KEYBOARD_KEYS:
            raise Exception(f"{char} is not a valid key to press")

        pyautogui.keyUp(char)

    def char_to_key(self, char):
        if char == ' ':
            intended_key = 'space'
        elif char == '\n':
            intended_key = 'return'
        elif char == '\t':
            intended_key = 'tab'
        else:
            intended_key = char.lower()
        return intended_key

    def _generate_typo(self, intended_key):
        needs_shift = '~!@#$%^&*()_+{}|:"<>?' + string.ascii_uppercase

        if intended_key == ' ':
            intended_key = 'space'
        elif intended_key == '\n':
            intended_key = 'return'
        elif intended_key == '\t':
            intended_key = 'tab'
        else:
            intended_key = intended_key.lower()

        # We generate a random displacement
        x_offset = np.random.randint(-1,2)
        y_offset = np.random.randint(-1,2)

        typo_coordinates = np.array(self._layout[intended_key]) - np.array((x_offset, y_offset))

        # Find posibilities
        typo_key = intended_key
        for k, v in self._layout.items():
            if np.all(np.array(v) == typo_coordinates) and k not in needs_shift:
                typo_key = k

        return typo_key

    def _generate_hold_delay_dict(self):
        ''' Calcula un diccionario de hold delay aleatorio
        '''
        hold_dict = {k: self._handicap * np.random.uniform(low=0.2,high=0.6) for k in self._layout.keys()}
        return hold_dict
    
    def _generate_move_delay_matrix(self):
        ''' Calcula una matriz de tiempo de vuelo aleatoria
        '''
        # We calculate a distance matrix for each combination of keys
        distance_matrix = np.empty((len(self._key_to_index), len(self._key_to_index)))
        for k_i, i in self._key_to_index.items():
            for k_j, j in self._key_to_index.items():
                # We calculate the distance between the two keys
                pos_i = np.array(self._layout[k_i])
                pos_j = np.array(self._layout[k_j])
                distance = np.linalg.norm(pos_i - pos_j)

                distance_matrix[i][j] = distance

        # Now we sum to the places where the destination is a letter, a scalar proportional to the handicap
        for k, i in self._key_to_index.items():
            if k not in string.ascii_letters:
                continue

            distance_matrix[:][i] += np.random.uniform(low=self._handicap, high=self._handicap+0.3)

        # Now we sum to the places where the destination is a number, a scalar proportional to the handicap
        for k, i in self._key_to_index.items():
            if k not in string.digits:
                continue

            distance_matrix[:][i] += np.random.uniform(low=self._handicap+0.1, high=self._handicap+0.4)

        # Now we sum to the places where the destination is a symbol, a scalar proportional to the handicap
        for k, i in self._key_to_index.items():
            if k not in string.punctuation.split() + ["return", "backspace", "ctrl", "shift", "capslock", "tab", "esc", "space"]:
                continue
            
            distance_matrix[:][i] += np.random.uniform(low=self._handicap+0.2, high=self._handicap+0.6) 

        return distance_matrix