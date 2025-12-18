import pandas as pd

def keep_relevant_features(data):
    vd = data["vehicleData"]
    return {
        "speed": vd["speed"],
        "time": vd["time"],
        "finished": vd["finished"],
        "distance_next_turn" : data["distance_next_turn"]
    }


def read_map():
    return pd.read_csv("Learning/utils/ParseTMMap/blocks.csv")

def find_block_index(car_pos, df):
    """
    Identifie le bloc (ligne du DataFrame) dans lequel se trouve la voiture,
    en utilisant uniquement les coordonnées de grille. Suppose une map simple

    car_pos : (x, y, z) coordonnées monde de la voiture (float)
    df      : DataFrame contenant les colonnes X, Y, Z (coordonnées grille)

    Retour :
        index du bloc dans df, ou None si aucun bloc ne correspond
    """

    car_x, car_y, car_z = car_pos

    # Conversion monde -> grille
    bx = int(car_x // 32)
    by = int(car_y // 8)
    bz = int(car_z // 32)

    # Recherche exacte du bloc
    match = df[
        (df["X"] == bx) &
        (df["Y"] == by) &
        (df["Z"] == bz)
    ]

    if match.empty:
        return None

    return match.index[0]

def next_curve_center_point_world_pos(current_idx, df):
    """
    Retourne (x, y, z) monde du début du prochain bloc Curve
    """

    for i in range(current_idx + 1, len(df)):
        if "Curve" in df.loc[i, "Block"]:
            return (
                df.loc[i, "cx"],
                df.loc[i, "cy"],
                df.loc[i, "cz"]
            )
        
    return None


def distance_3d(p1, p2):
    """
    calcule la distance entre deux points monde
    """
    if p1 == None or p2 == None:
        return float("inf")

    dx = p2[0] - p1[0]

    dz = p2[2] - p1[2]
    return (dx*dx + dz*dz) ** 0.5


def distance_to_next_turn(car_pos):
    """
    Calcule la distance entre la position actuel et l'entrée du prochain virage
    car_pos : (x, y, z) coordonnées monde de la voiture (float)
    df      : DataFrame contenant les colonnes X, Y, Z (coordonnées grille), typiquement généré par le code en C

    """
    # On lit les données
    df = read_map()

    # On cherche le bloc actuel
    current_idx = find_block_index(car_pos, df)
    if current_idx is None:
        return float("inf")

    # entrée du prochain virage
    curve_pos = next_curve_center_point_world_pos(current_idx, df)
    if curve_pos is None:
        return float("inf")

    distance_to_center = distance_3d(car_pos, curve_pos)
    distance_to_entry_of_turn = distance_to_center - 16

    return distance_to_entry_of_turn









