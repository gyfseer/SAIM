from matplotlib import font_manager

font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

for path in font_paths:
    try:
        name = font_manager.FontProperties(fname=path).get_name()
        print(f"{name:<40} -> {path}")

    except Exception as e:
        print("false")