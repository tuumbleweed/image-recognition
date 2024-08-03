#ifndef COLORS_H
#define COLORS_H

#include <string>

// Reset
const std::string RESET = "\033[0m";          // (⬜) Reset to default color

// Text colors
const std::string BLACK = "\033[30m";         // (⚫) Black
const std::string RED = "\033[31m";           // (🟥) Red
const std::string GREEN = "\033[32m";         // (🟩) Green
const std::string YELLOW = "\033[33m";        // (🟨) Yellow
const std::string BLUE = "\033[34m";          // (🟦) Blue
const std::string MAGENTA = "\033[35m";       // (🟪) Magenta
const std::string CYAN = "\033[36m";          // (🟦🟩) Cyan
const std::string WHITE = "\033[37m";         // (⬜) White

// Bright text colors
const std::string BRIGHT_BLACK = "\033[90m";  // (⚫) Bright Black (Grey)
const std::string GREY = "\033[90m";          // (⚫) Grey (alias for Bright Black)
const std::string BRIGHT_RED = "\033[91m";    // (🟥) Bright Red
const std::string BRIGHT_GREEN = "\033[92m";  // (🟩) Bright Green
const std::string BRIGHT_YELLOW = "\033[93m"; // (🟨) Bright Yellow
const std::string BRIGHT_BLUE = "\033[94m";   // (🟦) Bright Blue
const std::string BRIGHT_MAGENTA = "\033[95m";// (🟪) Bright Magenta
const std::string BRIGHT_CYAN = "\033[96m";   // (🟦🟩) Bright Cyan
const std::string BRIGHT_WHITE = "\033[97m";  // (⬜) Bright White

// Background colors
const std::string BG_BLACK = "\033[40m";      // (⚫) Black background
const std::string BG_RED = "\033[41m";        // (🟥) Red background
const std::string BG_GREEN = "\033[42m";      // (🟩) Green background
const std::string BG_YELLOW = "\033[43m";     // (🟨) Yellow background
const std::string BG_BLUE = "\033[44m";       // (🟦) Blue background
const std::string BG_MAGENTA = "\033[45m";    // (🟪) Magenta background
const std::string BG_CYAN = "\033[46m";       // (🟦🟩) Cyan background
const std::string BG_WHITE = "\033[47m";      // (⬜) White background

// Bright background colors
const std::string BG_BRIGHT_BLACK = "\033[100m";  // (⚫) Bright Black (Grey) background
const std::string BG_GREY = "\033[100m";          // (⚫) Grey background (alias for Bright Black)
const std::string BG_BRIGHT_RED = "\033[101m";    // (🟥) Bright Red background
const std::string BG_BRIGHT_GREEN = "\033[102m";  // (🟩) Bright Green background
const std::string BG_BRIGHT_YELLOW = "\033[103m"; // (🟨) Bright Yellow background
const std::string BG_BRIGHT_BLUE = "\033[104m";   // (🟦) Bright Blue background
const std::string BG_BRIGHT_MAGENTA = "\033[105m";// (🟪) Bright Magenta background
const std::string BG_BRIGHT_CYAN = "\033[106m";   // (🟦🟩) Bright Cyan background
const std::string BG_BRIGHT_WHITE = "\033[107m";  // (⬜) Bright White background

// Text styles
const std::string BOLD = "\033[1m";           // Bold text
const std::string UNDERLINE = "\033[4m";      // Underlined text
const std::string INVERSE = "\033[7m";        // Inverse (swap foreground and background colors)
const std::string HIDDEN = "\033[8m";         // Hidden text
const std::string STRIKETHROUGH = "\033[9m";  // Strikethrough text

#endif // COLORS_H
