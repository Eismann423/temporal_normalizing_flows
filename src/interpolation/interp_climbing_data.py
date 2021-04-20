import pandas as pd
import numpy as np
import os
from scipy.interpolate import griddata

path_source = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\raw\csv"
path_dest = r"C:\Users\Eisma\thesis\nationals_data\calcs\qual\processing\white_proc\interpolation\csv"

freq = .008
for subdir, dirs, files in os.walk(path_source):
    for filename in files:
        # Reads file and extracts data from each column
        xls = pd.read_csv(os.path.join(subdir, filename))
        x = np.array([xls.loc[:, 'X'].values])
        y = np.array([xls.loc[:, 'Y'].values])
        z = np.array([xls.loc[:, 'Z'].values])
        values = np.concatenate(x, y, z)

        grid_x, grid_y, grid_z = np.mgrid[0:1:100j, 0:1:200j]

        endTime = len(x)
        time = np.arange(0, endTime, freq)

        grid_w0 = griddata(time, values, (grid_x, grid_y), method='nearest')
        grid_w1 = griddata(time, values, (grid_x, grid_y), method='linear')
        grid_w2 = griddata(time, values, (grid_x, grid_y), method='cubic')

        # To Do
        # Exports interpolated data to csv
        sheet1 = pd.read_excel(xls, sheet_name="first")
        new_filename = os.path.join(path_dest,
                                    os.path.basename(subdir) + '\\' + os.path.splitext(filename)[0] + " first.csv")
        sheet1.to_csv(new_filename, columns=['X', 'Y', 'Z'], encoding='utf-8')