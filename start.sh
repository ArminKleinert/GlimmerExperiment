#clang -Wall -Wextra -pedantic -Os -fsanitize=address -fno-omit-frame-pointer -g -o experiment experiment.c && ./experiment
clang -Wall -Wextra -pedantic -O3 -o experiment experiment.c && ./experiment
rm experiment
