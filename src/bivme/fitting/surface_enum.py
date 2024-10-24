from enum import IntEnum, Enum


class Surface(IntEnum):
    """ """

    LV_ENDOCARDIAL = 0  # LV_ENDOCARDIAL,
    RV_SEPTUM = 1  # RV septum,
    RV_FREEWALL = 2  # RV free wall,
    EPICARDIAL = 3  # epicardium,
    MITRAL_VALVE = 4  # mitral valve,
    AORTA_VALVE = 5  # aorta,
    TRICUSPID_VALVE = 6  # tricuspid,
    PULMONARY_VALVE = 7  # pulmonary valve,
    RV_INSERT = 8  # RV insert.
    APEX = 9

class ControlMesh(IntEnum):
    """ """
    LV_ENDOCARDIAL = 0  # LV_ENDOCARDIAL,
    RV_ENDOCARDIAL = 1  # RV septum,
    EPICARDIAL = 2  # epicardium,

class ContourType(Enum):
    """ """

    # SAX = short axis
    # LAX = long axis
    SAX_RV_FREEWALL = "SAX_RV_FREEWALL"
    LAX_RV_FREEWALL = "LAX_RV_FREEWALL"
    SAX_RV_SEPTUM = "SAX_RV_SEPTUM"
    LAX_RV_SEPTUM = "LAX_RV_SEPTUM"
    SAX_RV_OUTLET = "SAX_RV_OUTLET"

    RV_INSERT = "RV_INSERT"

    LAX_RV_ENDOCARDIAL = "LAX_RV_ENDOCARDIAL"
    LAX_RV_EPICARDIAL = "LAX_RV_EPICARDIAL"
    LAX_LV_ENDOCARDIAL = "LAX_LV_ENDOCARDIAL"
    LAX_LV_EPICARDIAL = "LAX_LV_EPICARDIAL"

    SAX_RV_ENDOCARDIAL = "SAX_RV_ENDOCARDIAL"
    SAX_RV_EPICARDIAL = "SAX_RV_EPICARDIAL"
    SAX_LV_ENDOCARDIAL = "SAX_LV_ENDOCARDIAL"
    SAX_LV_EPICARDIAL = "SAX_LV_EPICARDIAL"

    MITRAL_VALVE = "MITRAL_VALVE"
    MITRAL_PHANTOM = "MITRAL_PHANTOM"
    APEX_POINT = "APEX_POINT"
    TRICUSPID_VALVE = "TRICUSPID_VALVE"
    PULMONARY_VALVE = "PULMONARY_VALVE"
    PULMONARY_PHANTOM = "PULMONARY_PHANTOM"
    AORTA_VALVE = "AORTA_VALVE"
    AORTA_PHANTOM = "AORTA_PHANTOM"
    TRICUSPID_PHANTOM = "TRICUSPID_PHANTOM"
    LAX_RA = "LAX_RA"
    LAX_LA = "LAX_LA"
    LAX_RV_EXTENT = "LAX_RV_EXTENT"
    LAX_LV_EXTENT = "LAX_LV_EXTENT"
    LAX_LA_EXTENT = "LAX_LA_EXTENT"
    SAX_RA = "SAX_RA"
    SAX_LA = "SAX_LA"
    SAX_LAA = "SAX_LAA"
    SAX_LPV = "SAX_LPV"
    SAX_RPV = "SAX_RPV"
    SAX_SVC = "SAX_SVC"
    SAX_IVC = "SAX_IVC"
    EXCLUDED = "EXCLUDED"

SURFACE_CONTOUR_MAP = {
    Surface.LV_ENDOCARDIAL.value: [
        ContourType.SAX_LV_ENDOCARDIAL,
        ContourType.LAX_LV_ENDOCARDIAL,
    ],
    Surface.RV_SEPTUM.value: [ContourType.LAX_RV_SEPTUM, ContourType.SAX_RV_SEPTUM],
    Surface.RV_FREEWALL.value: [
        ContourType.SAX_RV_FREEWALL,
        ContourType.LAX_RV_FREEWALL,
        ContourType.SAX_RV_OUTLET,
    ],
    Surface.EPICARDIAL.value: [
        ContourType.SAX_LV_EPICARDIAL,
        ContourType.LAX_LV_EPICARDIAL,
        ContourType.SAX_RV_EPICARDIAL,
        ContourType.LAX_RV_EPICARDIAL,
    ],
    Surface.MITRAL_VALVE.value: [ContourType.MITRAL_VALVE, ContourType.MITRAL_PHANTOM],
    Surface.AORTA_VALVE.value: [ContourType.AORTA_VALVE, ContourType.AORTA_PHANTOM],
    Surface.TRICUSPID_VALVE.value: [
        ContourType.TRICUSPID_VALVE,
        ContourType.TRICUSPID_PHANTOM,
    ],
    Surface.PULMONARY_VALVE.value: [
        ContourType.PULMONARY_VALVE,
        ContourType.PULMONARY_PHANTOM,
    ],
    Surface.RV_INSERT.value: [ContourType.RV_INSERT],
    Surface.APEX.value: [ContourType.APEX_POINT],
}
