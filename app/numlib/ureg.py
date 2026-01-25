import pint
import os

convertunits = {
    "Ah": "ampere*hour",
    "As2": "ampere*s**2",
    "Deg": "degC",
    "kg/m3": "kg/meter**3",
    "Kgm2": "kg*m**2",
    "kgm2": "kg*m**2",
    "kNm": "kN*m",
    "m/h": "meter/hour",
    "m/s2": "m/s**2",
    "m3/s": "m**3/s",
    "N/m2": "newton/m**2",
    "Nm": "newton*m",
    "Nm/A": "newton*meter/ampere",
    "Nm/rad": "newton*meter/rad",
    "Nms/rad": "newton*meter*sec/rad",
    "Ns/m2": "newton*sec/meter**2",
    "Ns/m3": "newton*sec/meter**3",
    "Ns/m4": "newton*sec/meter**4",
    "Ohm": "ohm",
    "PPR": "",
    "rad/Nms": "rad/(newton*meter*sec)",
    "rad/s2": "rad/sec**2",
    "rads2": "rad/sec**2",
    "RPM": "rpm",
    "v": "V",
    "Volt": "V",
    "Vs/rad": "volt*sec/rad",
    "%": "percent",
}

try:
    from defines import SETTINGSDIR

    units_definition_dir = SETTINGSDIR
except ImportError:
    SETTINGSDIR = None
    import sys

    if getattr(sys, "frozen", False):
        units_definition_dir = os.path.dirname(sys.executable)
    else:
        units_definition_dir = os.path.dirname(os.path.realpath(__file__))

print(units_definition_dir)
ureg = pint.UnitRegistry()
ureg.load_definitions(os.path.join(units_definition_dir, "pint.txt"))
ureg.default_format = "P~"
Q_ = ureg.Quantity
errors = pint.errors


# for pickle
def set_application_registry():
    pint.set_application_registry(ureg)
    # a= Q_(12,'snet_position')
    # print(a,type(a))
    print(f"*** Created new pint registry version: {pint.__version__} ***")
