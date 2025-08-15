# Breast Cancer Diagnosis AI Model

## Breast Cancer Wisconsin Dataset - Data Dictionary

### Patient Identification
- **id**: Unique patient identifier number
- **diagnosis**: **TARGET VARIABLE** - Tumor (ورم) diagnosis (M=Malignant (خبيث), B=Benign)

### Cell Nucleus Measurements - Mean Values
- **radius_mean**: Average radius of cell nucleus (pixels)
- **texture_mean**: Average texture variation in cell nucleus (gray-scale values)
- **perimeter_mean**: Average perimeter of cell nucleus (pixels)
- **area_mean**: Average area of cell nucleus (square pixels)
- **smoothness_mean**: Average smoothness of cell nucleus (variation in radius lengths)
- **compactness_mean**: Average compactness of cell nucleus (perimeter²/area - 1.0)
- **concavity_mean**: Average concavity of cell nucleus (severity of concave portions)
- **concave points_mean**: Average number of concave portions of cell nucleus
- **symmetry_mean**: Average symmetry of cell nucleus
- **fractal_dimension_mean**: Average fractal dimension of cell nucleus

### Cell Nucleus Measurements - Standard Error Values
- **radius_se**: Standard error of radius measurements
- **texture_se**: Standard error of texture measurements
- **perimeter_se**: Standard error of perimeter measurements
- **area_se**: Standard error of area measurements
- **smoothness_se**: Standard error of smoothness measurements
- **compactness_se**: Standard error of compactness measurements
- **concavity_se**: Standard error of concavity measurements
- **concave points_se**: Standard error of concave points measurements
- **symmetry_se**: Standard error of symmetry measurements
- **fractal_dimension_se**: Standard error of fractal dimension measurements

### Cell Nucleus Measurements - Worst Values
- **radius_worst**: Largest radius value found in cell nucleus
- **texture_worst**: Largest texture value found in cell nucleus
- **perimeter_worst**: Largest perimeter value found in cell nucleus
- **area_worst**: Largest area value found in cell nucleus
- **smoothness_worst**: Largest smoothness value found in cell nucleus
- **compactness_worst**: Largest compactness value found in cell nucleus
- **concavity_worst**: Largest concavity value found in cell nucleus
- **concave points_worst**: Largest number of concave points found in cell nucleus
- **symmetry_worst**: Largest symmetry value found in cell nucleus
- **fractal_dimension_worst**: Largest fractal dimension value found in cell nucleus

### Data Quality Note
- **Unnamed: 32**: Empty column with no useful information (should be removed)

---

## What These Measurements Mean

### Key Concepts:
- **Mean values**: Average measurements across all cells in the sample
- **Standard error (se)**: How much the measurements vary from the average
- **Worst values**: The most extreme (highest) measurements found

### Why These Features Matter:
- **Larger measurements** (radius, area, perimeter) often indicate **malignant** tumors
- **Higher texture values** suggest more irregular cell surfaces (common in cancer)
- **Lower smoothness** means more irregular cell shapes (cancer indicator)
- **Higher concavity** indicates more irregular, "jagged" cell boundaries
- **Lower symmetry** suggests abnormal, asymmetric cell shapes

### Medical Context:
- **Benign tumors**: Usually have smaller, more regular measurements
- **Malignant tumors**: Usually have larger, more irregular measurements
- **Standard errors**: Show how consistent the measurements are within the sample
- **Worst values**: Capture the most concerning aspects of the tumor
