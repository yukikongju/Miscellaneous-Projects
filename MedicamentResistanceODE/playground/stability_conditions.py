""" 
Stability Points values

https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch2/portrait.html
"""

# values: Docetaxel
lWT = 0.4756
lM = 0.3964
KWT = 250
KM = 1e6
aWT = 0.0015
aM = -0.0004

# values: Afatinib
lWT = 0.5487
lM = 0.0132
KWT = 1000
KM = 28.82
aWT = -0.0007
aM = -0.0049


# TRIVIAL: (0,0)
print("Point: (0,0)")
det = lM * lWT
tr = lM + lWT
discr = (lM - lWT)**2
print(f"Det: {det}")
print(f"Trace: {tr}")
print(f"Discriminant: {discr}")
print()

# (KWT, 0)
print("Point: (KWT,0)")
det = -(1+aM * KWT) * lM * lWT
tr = lM + aM * KWT * lM - lWT
discr = (lM + aM * KWT * lM - lWT)**2
print(f"Det: {det}")
print(f"Trace: {tr}")
print(f"Discriminant: {discr}")
print()

# (0, KM)
print("Point: (0,KM)")
det = -(1+aM * KWT) * lM * lWT
tr = -lM + aWT * KM * lWT + lWT
discr = (lM + lWT + aWT * KM * lWT)**2
print(f"Det: {det}")
print(f"Trace: {tr}")
print(f"Discriminant: {discr}")
print()

# Non-trivial
print("Point: Non trivial point")
det = -((1 + aWT * KWT) * (1 + aM * KWT) * lM * lM) / (-1 + aM * aWT * KWT)
tr = (lM + aM * KWT * lM + lM + aWT * KM * lWT) / (-1 + aM * aWT * KWT)
discr = ((lM + aM * KWT * lM)**2 + 2 * (1 + aWT * KM) * (1 + aM * KWT) * (-1 + 2 * aM * aWT * KM * KWT) * lM * lWT + (lWT + aWT * KM * lWT)**2) / (-1 + aM * aWT * KM * KWT)**2
print(f"Det: {det}")
print(f"Trace: {tr}")
print(f"Discriminant: {discr}")
print()
