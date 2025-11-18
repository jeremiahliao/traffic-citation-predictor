import difflib
import re
import pandas as pd

CATEGORICAL_COLS = [
    # driver information
    'VehicleType',

    # traffic stop information
    'Charge', 'Arrest Type',
]
NUMERICAL_COLS = [
    # binary columns
    'Accident', 'Belts', 'Personal Injury', 'Property Damage', 'Fatal',
    'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol',
    'Work Zone', 'Contributed To Accident'
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

canonical_makes = [
    "ACURA","ALFA ROMEO","ASTON MARTIN","AUDI","BENTLEY","BMW","BUGATTI","BUICK",
    "CADILLAC","CHEVROLET","CHRYSLER","DODGE","FERRARI","FIAT","FORD","GENESIS","GMC",
    "HONDA","HUMMER","HYUNDAI","INFINITI","ISUZU","JAGUAR","JEEP","KIA","LAMBORGHINI",
    "LAND ROVER","LEXUS","LINCOLN","MASERATI","MAZDA","MERCEDES-BENZ","MERCURY","MINI",
    "MITSUBISHI","NISSAN","OLDSMOBILE","PEUGEOT","PLYMOUTH","PONTIAC","PORSCHE","RAM",
    "RENAULT","ROLLS-ROYCE","SAAB","SATURN","SCION","SMART","SUBARU","SUZUKI","TESLA",
    "TOYOTA","VOLKSWAGEN","VOLVO",

    "MACK","FREIGHTLINER","PETERBILT","KENWORTH","HINO","INTERNATIONAL",
    "HARLEY DAVIDSON","KAWASAKI","YAMAHA",

    "NONE", "NA", 
]

alias_map = {
    # --- Toyota ---
    "TOYOT":"TOYOTA","TOYTA":"TOYOTA","TOYO":"TOYOTA","TOYT":"TOYOTA",
    "TOTY":"TOYOTA","TYT":"TOYOTA",

    # --- Hyundai ---
    "HYANDAI":"HYUNDAI","HYUNDIA":"HYUNDAI","HYUNDI":"HYUNDAI","HYND":"HYUNDAI",

    # --- Mercedes-Benz ---
    "MERC":"MERCEDES-BENZ","MERZ":"MERCEDES-BENZ","MERZ BENZ":"MERCEDES-BENZ",
    "MB":"MERCEDES-BENZ","M B":"MERCEDES-BENZ","M-B":"MERCEDES-BENZ","BENZ":"MERCEDES-BENZ",
    "MERCEDES":"MERCEDES-BENZ","MERCEDES BENZ":"MERCEDES-BENZ","MERCDES":"MERCEDES-BENZ",

    # --- Chevrolet ---
    "CHEV":"CHEVROLET","CHEVY":"CHEVROLET","CHEVEROLET":"CHEVROLET","CHV":"CHEVROLET",
    "CHE":"CHEVROLET",

    # --- Infiniti ---
    "INF":"INFINITI","INFINITY":"INFINITI","INFIN":"INFINITI",

    # --- Lexus ---
    "LEX":"LEXUS","LEXSUS":"LEXUS","LEXS":"LEXUS",

    # --- Mitsubishi ---
    "MITS":"MITSUBISHI","MITZ":"MITSUBISHI","MITUSBISHI":"MITSUBISHI","MISUBISHI":"MITSUBISHI",
    "MITUBISHI":"MITSUBISHI","MITUSB":"MITSUBISHI",

    # --- Oldsmobile ---
    "OLDS":"OLDSMOBILE","OLDSMOB":"OLDSMOBILE","OLDSM":"OLDSMOBILE",

    # --- Land Rover ---
    "LNDR":"LAND ROVER","RANG":"LAND ROVER","RANGE":"LAND ROVER","RANGE ROVER":"LAND ROVER",
    "LANDROVER":"LAND ROVER",

    # --- Dodge / Chrysler ---
    "DODG":"DODGE","CRYSLER":"CHRYSLER","CHRSYLER":"CHRYSLER","CHR":"CHRYSLER",

    # --- Honda ---
    "HOND":"HONDA","HON":"HONDA",

    # --- Volkswagen ---
    "VW":"VOLKSWAGEN","VOLKS":"VOLKSWAGEN","VOLKSWAGON":"VOLKSWAGEN","WOLKSWAGEN":"VOLKSWAGEN",
    "VOLLKSWAGEN":"VOLKSWAGEN","VOLKW":"VOLKSWAGEN",

    # --- Cadillac ---
    "CAD":"CADILLAC","CADI":"CADILLAC","CADDY":"CADILLAC",

    # --- Nissan ---
    "NIS":"NISSAN","NISN":"NISSAN","NISS":"NISSAN",

    # --- Subaru ---
    "SUB":"SUBARU","SUBA":"SUBARU",

    # --- Ford ---
    "FRD":"FORD","FD":"FORD",

    # --- GMC ---
    "GM":"GMC","G M C":"GMC",

    # --- Mazda ---
    "MAZD":"MAZDA",

    # --- Genesis ---
    "GEN":"GENESIS","GENSIS":"GENESIS",

    # --- Porsche ---
    "PORSHE":"PORSCHE","POR":"PORSCHE",

    # --- Tesla ---
    "TSLA":"TESLA",

    # --- Harley-Davidson ---
    "HARLEY":"HARLEY DAVIDSON","HARLEY-DAVIDSON":"HARLEY DAVIDSON","HARL":"HARLEY DAVIDSON",

    # --- Yamaha ---
    "YAMA":"YAMAHA","YAM":"YAMAHA",

    # --- Kawasaki ---
    "KAW":"KAWASAKI","KAWK":"KAWASAKI",

    # --- Truck / Heavy vehicle makes ---
    "MACK":"MACK",
    "INTL":"INTERNATIONAL","INTERNATIONAL":"INTERNATIONAL",
    "FRHT":"FREIGHTLINER","FRTLN":"FREIGHTLINER","FREIGHT":"FREIGHTLINER",
    "PETE":"PETERBILT","PETERBUILT":"PETERBILT",
    "KEN":"KENWORTH","KW":"KENWORTH",
    "HINO":"HINO",

    # --- Misc abbreviations ---
    "BM":"BMW","B M W":"BMW","B.M.W":"BMW","BMW":"BMW",
    "VOLVO":"VOLVO","VOLV":"VOLVO","VO":"VOLVO",
    "JAG":"JAGUAR","JAGR":"JAGUAR",
    "LEXUS":"LEXUS","INFINITI":"INFINITI",
    "ACUR":"ACURA","ACURA":"ACURA",
    "HYUNDAI":"HYUNDAI","HONDA":"HONDA","FORD":"FORD",
}

charge_to_description_map = {
    '21-1124.1(b)': 'DRIVER READING A ELECTRONIC MSG. WHILE OPER. VEH. IN TRAVEL PORTION OF HWY',
    '22-412.3(b)': 'OCCUPANT UNDER 16 NOT RESTRAINED BY SEATBELT',
    '21-902(c1)': 'DRIVING VEH. WHILE SO FAR IMPAIRED BY DRUGS AND ALCOHOL CANNOT DRIVE SAFELY',
    '21-209(3)': 'FAILING TO STOP UNTIL SAFE TO CONTINUE THROUGH INTERSECTION W/NONFUNCT. TRAF. SIGNAL',
    '21-801(a)': 'DRIVING VEHICLE IN EXCESS OF REASONABLE AND PRUDENT SPEED ON HIGHWAY',
    '21-209(3)': 'FAILING TO STOP UNTIL SAFE TO CONTINUE THROUGH INTERSECTION W/NONFUNCT. TRAF. SIGNAL',
    '16-101(a)': 'DRIVING MOTOR VEHICLE ON HIGHWAY WITHOUT REQUIRED LICENSE AND AUTHORIZATION',
    '11-393.78': 'OPERATING MOTOR VEHICLE WITH INADEQUATE WINDSHIELD WIPERS-no fluid',
    '13-411(f)': 'DISPLAYING EXPIRED REGISTRATION PLATE ISSUED BY ANY STATE',
}

def normalize_make(x: str) -> str:
    if not isinstance(x, str):
        return ""
    s = x.upper().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def resolve_make(raw: str) -> str:
    s = normalize_make(raw)
    if not s:
        return "UNKNOWN"
    if s in alias_map:
        return alias_map[s]
    if s in canonical_makes:
        return s
    first = s.split(" ")[0]
    if first in alias_map:
        return alias_map[first]
    match = difflib.get_close_matches(s, canonical_makes, n=1, cutoff=0.65)
    if match:
        return match[0]
    match = difflib.get_close_matches(first, canonical_makes, n=1, cutoff=0.65)
    return match[0] if match else "UNKNOWN"

if __name__ == '__main__':
    cols = [
        'Date Of Stop', 'Time Of Stop', 'Agency', 'SubAgency', 'Description',
       'Location', 'Latitude', 'Longitude', 'Accident', 'Belts',
       'Personal Injury', 'Property Damage', 'Fatal', 'Commercial License',
       'HAZMAT', 'Commercial Vehicle', 'Alcohol', 'Work Zone', 'State',
       'VehicleType', 'Year', 'Make', 'Model', 'Color', 'Violation Type',
       'Charge', 'Article', 'Contributed To Accident', 'Race', 'Gender',
       'Driver City', 'Driver State', 'DL State', 'Arrest Type',
       'Geolocation'
    ]
    df = pd.read_csv('Traffic_Violations.csv')

    df['Make'] = df['Make'].apply(resolve_make)

    # fill in missing description
    df['Description'] = df['Description'].fillna(df['Charge'].map(charge_to_description_map))

    # inconsistency as to which state value is filled out, create one uniform column
    df['State'] = df[['State', 'DL State', 'Driver State']].bfill(axis=1).iloc[:, 0]

    # fill in NaN with unknowns
    df['Model'] = df['Model'].fillna('Unknown')
    df['Color'] = df['Color'].fillna('Unknown')

    # set NaN coordinates to center of Montomery County
    df['Latitude'] = df['Latitude'].fillna(39.1547)
    df['Longitude'] = df['Longitude'].fillna(-77.2405)

    binary_cols = [
        'Accident', 'Belts', 'Personal Injury', 'Property Damage', 'Fatal',
        'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol',
        'Work Zone', 'Contributed To Accident'
    ]
    df[binary_cols] = (df[binary_cols] == "Yes").astype(int)

    # keep only data with warnings and citations
    df = df[(df['Violation Type'] == 'Warning') | (df['Violation Type'] == 'Citation')]

    df.to_parquet('Traffic_Violations.parquet', index=False)