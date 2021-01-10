import numpy as np

class Occupation:
    def __init__(self, priority, time, probability, start_time = None):
        self.priority = priority
        self.time = int(time)
        self.probability = probability
        self.start_time = int(start_time) if start_time is not None else None

def generate_scheduling_function(person_seed,
                                max_day_var=0.1,
                                sleep_activation_prob = 0.02,
                                high_occupation_prob = 0.1,
                                mid_occupation_prob = 0.3,
                                low_occupation_prob = 0.6):
    ''' Retorna una funcion con el prototipo `day_func(day_seed, amount = 1)`
    para generar en que minuto/s se debe activar el bot.
    `person_seed`: Seed del bot para generar su horario \n
    `max_day_var`: Maxima variacion por dia \n
    `sleep_activation_prob`: Probabilidad de que se active durmiendo \n
    `high_occupation_prob`: Probabilidad de que se active estando muy ocupado \n
    `mid_occupation_prob`: Probabilidad de que se active estando medianamente ocupado \n
    `low_occupation_prob`: Probabilidad de que se active estando poco ocupado
    '''
    # TODO: HEAVILY OPTIMICE

    def hour2min(hours):
        return int(hours * 60)

    def min2hour(sec):
        return sec/(60)

    ## First of all we generate the normal values for the person seed
    np.random.seed(person_seed)
    remaining_time = hour2min(24)

    # Sleep
    sleep_time = np.random.triangular(hour2min(6), hour2min(8), hour2min(12))
    remaining_time -= sleep_time

    # Work
    work_time = np.random.triangular(hour2min(5), hour2min(7), hour2min(8))
    remaining_time -= work_time

    # Chores
    total_chores_time = np.random.triangular(hour2min(1), hour2min(3), hour2min(min2hour(remaining_time) - 1))
    remaining_time -= total_chores_time

    # Breaks
    total_free_breaks_time = np.random.triangular(hour2min(1), hour2min(2), hour2min(min2hour(remaining_time) - 1))
    remaining_time -= total_free_breaks_time
    np.random.seed()
    
    ## Generate function

    def day_func(day_seed, amount = 1):
        # Step 1: Generate day variations
        np.random.seed(day_seed)

        # Sleep
        sleep_var = sleep_time * max_day_var * ((np.random.rand()-0.5)/0.5)

        # Work
        work_var = work_time * max_day_var * ((np.random.rand()-0.5)/0.5)

        # Chores
        chores = np.random.rand(np.random.randint(3,high=7))
        chores /= np.sum(chores)
        chores *= total_chores_time

        # Random Breaks
        free_breaks = np.random.rand(np.random.randint(8,high=15))
        free_breaks /= np.sum(free_breaks)
        free_breaks *= total_free_breaks_time

        np.random.seed()
        
        # Step 1: Generate occupations
        day_occupations = []
        # Add Sleep
        day_occupations += [Occupation(1, sleep_time + sleep_var, sleep_activation_prob, 0)]

        # Add Work
        day_occupations += [Occupation(3, work_time + work_var, high_occupation_prob)]

        # Add chores
        for c in chores:
            # set random start time
            start_time = np.random.triangular(sleep_time + sleep_var, hour2min(19), hour2min(24) - c)
            day_occupations += [Occupation(2, c, mid_occupation_prob, start_time=start_time)]

        # Add random breaks
        for f in free_breaks:
            # set random start time
            start_time = np.random.triangular(sleep_time + sleep_var, hour2min(14), hour2min(24) - f)
            day_occupations += [Occupation(0, f, low_occupation_prob, start_time=start_time)]

        day_occupations.sort(key=lambda x: x.priority)

        # Step 3: Generate distribution
        current_priority = 9999
        current_prob = low_occupation_prob

        day_minutes = []
        for m in range(hour2min(24)):
            for i, o in enumerate(day_occupations):
                # Delete already consumed occupations
                if o.time <= 0:
                    day_occupations.pop(i)

                # Take on occupations with a bigger priority
                if o.priority < current_priority:
                    if o.start_time is None:
                        current_occupation = o

                    # Take on occupations with a bigger priority that start in  that time
                    elif o.start_time == m:
                        current_occupation = o
                        break

            if current_occupation is not None:
                current_occupation.time -= 1
                if current_occupation.time <= 0:
                    current_occupation = None

            current_priority = current_occupation.priority if current_occupation is not None else 9999
            current_prob = current_occupation.probability if current_occupation is not None else low_occupation_prob

            day_minutes.append(current_prob)

        day_minutes = np.array(day_minutes)
        # Now that we have a probability for each minute, normalize the values so the min val is 1
        min_prob = np.min([sleep_activation_prob, low_occupation_prob, mid_occupation_prob, high_occupation_prob])

        day_minutes = day_minutes // min_prob

        # Step 4: Generate random numbers FIXME
        total = np.sum(day_minutes)

        results = []
        for _ in range(amount):
            rand_amount = np.random.randint(1, total)\
            for i in range(len(day_minutes)):
                # Tomamos las papeletas de ese min
                count = day_minutes[i]
                
                while count > 0:
                    rand_amount -= 1
                    count -= 1

                if rand_amount < 0:
                    results.append(i)
                    break

        return results[0] if len(results) == 1 else results

    return day_func
