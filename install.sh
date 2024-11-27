#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CHECK_MARK="\xE2\x9C\x94"
CROSS_MARK="\xE2\x9D\x8C"
WRENCH="\xF0\x9F\x94\xA7"
PACKAGE="\xF0\x9F\x93\xA6"
ROCKET="\xF0\x9F\x9A\x80"
CHAIN="\xE2\x9B\x93\xEF\xB8\x8F"
PUZZLE="\xF0\x9F\xA7\xA9"

MERIDIANS="\xF0\x9F\x8C\x90"
GEAR="\xE2\x9A\x99"
DIRECTORY="\xF0\x9F\x93\x81"
TRASH="\xF0\x9F\x97\x91"
LINK="\xF0\x9F\x94\x97"

VERBOSE=0
if [[ "$1" == "--verbose" || "$1" == "-v" ]]; then
    VERBOSE=1
fi

execute_command() {
    echo -e "${3}${NC} $2${NC}"
    if [ $VERBOSE -eq 1 ]; then
        $1
    else
        $1 > /dev/null 2>&1
    fi
}

echo -e "${LINK}${NC} Installing the ${GREEN}custatevec${NC} and ${GREEN}lbfgs${NC} libraries${NC}"
conda install custatevec conda-forge::liblbfgs
execute_command "pip install maturin[patchelf]" "Installing the ${GREEN}maturin${NC} build tool with ${GREEN}patchelf${NC}" "$PUZZLE"
execute_command "maturin build --release" "Building the project from source... (this might take a while)" "$GEAR "

WHEEL_FILE=$(find target/wheels -name '*.whl' | sort | tail -n 1)

if [[ ! -z "$WHEEL_FILE" ]]; then
    execute_command "pip install $WHEEL_FILE" "Installing the wheel file with pip..." "$PACKAGE"
    echo -e "${GREEN}${CHECK_MARK}  Installation completed successfully. ${NC}"
else
    echo -e "${RED}${CROSS_MARK} Wheel file not found.${NC}"
    exit 1
fi
