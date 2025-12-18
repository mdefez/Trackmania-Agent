import pandas as pd


def read_map():
    return pd.read_csv("Learning/ParseTMMap/blocks.csv", sep = ";", encoding="utf-16")

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

def next_curve_entry_world_pos(current_idx, df):
    """
    Retourne (x, y, z) monde du début du prochain bloc Curve
    """

    for i in range(current_idx + 1, len(df)):
        if "Curve" in df.loc[i, "Block"]:
            bx = df.loc[i, "X"]
            by = df.loc[i, "Y"]
            bz = df.loc[i, "Z"]

            x = bx * 32
            y = by * 8
            z = bz * 32

            return (x, y, z)

    return None


def distance_3d(p1, p2):
    """
    calcule la distance entre deux points monde
    """
    if p1 == None or p2 == None:
        return float("inf")

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    return (dx*dx + dy*dy + dz*dz) ** 0.5


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
    curve_pos = next_curve_entry_world_pos(current_idx, df)
    if curve_pos is None:
        return float("inf")

    return distance_3d(car_pos, curve_pos)



print(distance_to_next_turn((16.1*32, 9*8, 21.1*32)))







