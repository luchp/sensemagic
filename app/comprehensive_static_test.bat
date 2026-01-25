@echo off
REM comprehensive_static_test.bat - Comprehensive test of static files from Windows

echo ==========================================
echo COMPREHENSIVE STATIC FILES TEST
echo ==========================================
echo.

echo Step 1: Testing all static files...
echo.

echo ----------------------------------------
echo CSS Files:
echo ----------------------------------------
echo.
echo 1. base.css:
curl -I https://www.sensemagic.nl/static/css/base.css 2>&1 | findstr /C:"HTTP" /C:"Content-Length" /C:"Cache-Control"
echo.

echo 2. standalone.css:
curl -I https://www.sensemagic.nl/static/css/standalone.css 2>&1 | findstr /C:"HTTP" /C:"Content-Length" /C:"Cache-Control"
echo.

echo ----------------------------------------
echo JavaScript Files:
echo ----------------------------------------
echo.
echo 3. iframe-resize.js:
curl -I https://www.sensemagic.nl/static/js/iframe-resize.js 2>&1 | findstr /C:"HTTP" /C:"Content-Length"
echo.

echo 4. plot-updater.js:
curl -I https://www.sensemagic.nl/static/js/rectifier/plot-updater.js 2>&1 | findstr /C:"HTTP" /C:"Content-Length"
echo.

echo 5. form-utils.js:
curl -I https://www.sensemagic.nl/static/js/rectifier/form-utils.js 2>&1 | findstr /C:"HTTP" /C:"Content-Length"
echo.

echo ----------------------------------------
echo Image Files:
echo ----------------------------------------
echo.
echo 6. Rectifier schematic image:
curl -I https://www.sensemagic.nl/static/images/rectifier/single_phase_bridge_rectifier_schematic.jpg 2>&1 | findstr /C:"HTTP" /C:"Content-Length"
echo.

echo ==========================================
echo Step 2: Detailed content size tests...
echo ==========================================
echo.

echo Downloading and checking file sizes...
echo.

REM Create temp directory for downloads
set TEMP_DIR=%TEMP%\sensemagic_test
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo 1. base.css:
curl -s -o "%TEMP_DIR%\base.css" https://www.sensemagic.nl/static/css/base.css 2>nul
if exist "%TEMP_DIR%\base.css" (
    for %%A in ("%TEMP_DIR%\base.css") do (
        set SIZE=%%~zA
        echo    Size: %%~zA bytes ^(expected ~1741^)
        if %%~zA GTR 1000 (
            echo    [32m✓ SUCCESS[0m
        ) else (
            echo    [31m✗ FAILED - File too small[0m
        )
    )
) else (
    echo    [31m✗ FAILED - Could not download[0m
)
echo.

echo 2. standalone.css:
curl -s -o "%TEMP_DIR%\standalone.css" https://www.sensemagic.nl/static/css/standalone.css 2>nul
if exist "%TEMP_DIR%\standalone.css" (
    for %%A in ("%TEMP_DIR%\standalone.css") do (
        echo    Size: %%~zA bytes ^(expected ~1396^)
        if %%~zA GTR 1000 (
            echo    [32m✓ SUCCESS[0m
        ) else (
            echo    [31m✗ FAILED - File too small[0m
        )
    )
) else (
    echo    [31m✗ FAILED - Could not download[0m
)
echo.

echo 3. iframe-resize.js:
curl -s -o "%TEMP_DIR%\iframe-resize.js" https://www.sensemagic.nl/static/js/iframe-resize.js 2>nul
if exist "%TEMP_DIR%\iframe-resize.js" (
    for %%A in ("%TEMP_DIR%\iframe-resize.js") do (
        echo    Size: %%~zA bytes
        if %%~zA GTR 200 (
            echo    [32m✓ SUCCESS[0m
        ) else (
            echo    [31m✗ FAILED - File too small[0m
        )
    )
) else (
    echo    [31m✗ FAILED - Could not download[0m
)
echo.

echo 4. Rectifier schematic image:
curl -s -o "%TEMP_DIR%\schematic.jpg" https://www.sensemagic.nl/static/images/rectifier/single_phase_bridge_rectifier_schematic.jpg 2>nul
if exist "%TEMP_DIR%\schematic.jpg" (
    for %%A in ("%TEMP_DIR%\schematic.jpg") do (
        echo    Size: %%~zA bytes ^(expected ~45282^)
        if %%~zA GTR 10000 (
            echo    [32m✓ SUCCESS[0m
        ) else (
            echo    [31m✗ FAILED - File too small[0m
        )
    )
) else (
    echo    [31m✗ FAILED - Could not download[0m
)
echo.

echo ==========================================
echo Step 3: Testing actual page with static content...
echo ==========================================
echo.

echo Testing math documentation page ^(should load CSS and images^):
curl -I https://www.sensemagic.nl/app_rectifier/math?standalone=true 2>&1 | findstr /C:"HTTP" /C:"Content-Type"
echo.

echo ==========================================
echo Step 4: Full response headers for CSS...
echo ==========================================
echo.
curl -I https://www.sensemagic.nl/static/css/base.css
echo.

echo ==========================================
echo SUMMARY
echo ==========================================
echo.

REM Count successful downloads
set SUCCESS_COUNT=0
if exist "%TEMP_DIR%\base.css" (
    for %%A in ("%TEMP_DIR%\base.css") do if %%~zA GTR 1000 set /A SUCCESS_COUNT+=1
)
if exist "%TEMP_DIR%\standalone.css" (
    for %%A in ("%TEMP_DIR%\standalone.css") do if %%~zA GTR 1000 set /A SUCCESS_COUNT+=1
)
if exist "%TEMP_DIR%\iframe-resize.js" (
    for %%A in ("%TEMP_DIR%\iframe-resize.js") do if %%~zA GTR 200 set /A SUCCESS_COUNT+=1
)
if exist "%TEMP_DIR%\schematic.jpg" (
    for %%A in ("%TEMP_DIR%\schematic.jpg") do if %%~zA GTR 10000 set /A SUCCESS_COUNT+=1
)

echo Files successfully retrieved: %SUCCESS_COUNT% / 4
echo.

if %SUCCESS_COUNT% EQU 4 (
    echo [32m
    echo ==========================================
    echo ✓✓✓ ALL TESTS PASSED! ✓✓✓
    echo ==========================================
    echo.
    echo Static files are working correctly!
    echo Your images, CSS, and JavaScript should all load.
    echo [0m
) else (
    echo [31m
    echo ==========================================
    echo ✗✗✗ SOME TESTS FAILED ✗✗✗
    echo ==========================================
    echo.
    echo Static files are NOT working correctly.
    echo [0m
    echo.
    echo Look at the HTTP status codes above:
    echo   - 200 OK = Working
    echo   - 403 Forbidden = Permission issue
    echo   - 404 Not Found = File or config issue
    echo.
    echo On the Linux server, run:
    echo   cd /home/projects/sensemagic/app
    echo   chmod +x final_static_test.sh
    echo   ./final_static_test.sh
)
echo.

REM Clean up temp files
rmdir /S /Q "%TEMP_DIR%" 2>nul

echo ==========================================
echo Press any key to exit...
pause >nul

