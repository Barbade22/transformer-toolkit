class C:
    RESET = "\033[0m";  BOLD  = "\033[1m";  DIM    = "\033[2m"
    GREEN = "\033[32m"; CYAN  = "\033[36m"; YELLOW = "\033[33m"
    BLUE  = "\033[34m"; RED   = "\033[31m"; WHITE  = "\033[37m"
    MAGENTA = "\033[35m"

def _bar(current, total, width=28):
    filled = int(width * current / max(total, 1))
    return f"{C.CYAN}{'█' * filled}{'░' * (width - filled)}{C.RESET}"

def _section(title):
    print(f"\n{C.BOLD}{C.CYAN}{'─' * 52}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─' * 52}{C.RESET}")

def _info(label, value):
    print(f"  {C.DIM}{label:<18}{C.RESET} {C.WHITE}{value}{C.RESET}")

def _ok(msg):
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")

def _err(msg):
    print(f"  {C.RED}✗{C.RESET}  {msg}")