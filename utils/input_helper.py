


class InputHelper:
    '''
    Manages user input
    '''

    def __init__(self, interactive=True):
        '''
        Creates the input helper

        :param interactive: If true it is not needed to press ENTER after issuing each action
        '''
        import sys, tty
        self.interactive = interactive

    def read_character(self) -> str:
        '''
        Reads the next character input by the user
        :return: the read character
        '''
        import sys, tty, termios

        if self.interactive:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            return input()