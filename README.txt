python window_deformation.py                          # single case -> figures/*_clamped.jpg + single_case_clamped.xlsx
python window_deformation.py --sweep-variable diameter_mm   # 250-450 mm sweep -> jpg + xlsx (boundary condition appended)
python window_deformation.py --sweep-variable all      # combined thickness+diameter -> sweep_combined_clamped.jpg + sweep_combined_clamped.xlsx
python window_deformation.py --no-excel                # figures only, no spreadsheets
python window_deformation.py --no-figures              # spreadsheets only, no figures
python window_deformation.py --no-save                 # console output only, no files
python window_deformation.py --diameter-mm 250 --thickness-mm 35 --plate-theory mindlin  # shear-deformable thick plate (auto picks this when t/D > 0.1)
