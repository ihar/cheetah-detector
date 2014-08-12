rem pyinstaller console-detector.py --onefile --hidden-import=sklearn.utils.sparsetools._graph_validation
rem console-detector.exe d:/inputlist.txt d:/out.csv d:/codebook.pkl d:/gtb.pkl
rmdir /s /q build
rmdir /s /q dist

pyinstaller console-detector.py ^
            --hidden-import=sklearn.cluster.k_means_ ^
            --hidden-import=sklearn.utils.sparsetools._graph_validation ^
            --hidden-import=sklearn.utils.sparsetools._graph_tools ^
            --hidden-import=scipy.special._ufuncs_cxx ^
            --hidden-import=sklearn.utils.lgamma ^
            --hidden-import=sklearn.neighbors.typedefs ^
            --hidden-import=sklearn.utils.weight_vector ^
            --hidden-import=sklearn.ensemble.gradient_boosting ^
            --hidden-import=sklearn.ensemble.base ^
            --hidden-import=sklearn.ensemble.forest ^
            --hidden-import=sklearn.tree.tree ^
            --hidden-import=sklearn.tree._tree ^
            --hidden-import=sklearn.tree._utils ^
            --hidden-import=sklearn.tree.export ^
            --hidden-import=sklearn.ensemble.weight_boosting ^
            --hidden-import=sklearn.ensemble.bagging ^
            --hidden-import=numpy.core.umath_tests ^
            --hidden-import=sklearn.ensemble._gradient_boosting ^
            --hidden-import=sklearn.ensemble.partial_dependence ^
            --hidden-import=numbers ^
            
cp d:\Playground\python\cheetah-detector\application\console-detector\gtb.pkl d:\Playground\python\cheetah-detector\application\console-detector\dist\console-detector\gtb.pkl
cp d:\Playground\python\cheetah-detector\application\console-detector\codebook.pkl d:\Playground\python\cheetah-detector\application\console-detector\dist\console-detector\codebook.pkl

d:\Playground\python\cheetah-detector\application\console-detector\dist\console-detector\console-detector.exe d:\inputlist.txt d:\out.csv
            
pause