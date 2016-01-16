#!/usr/bin/env python
"""\
SVG.py - Construct/display SVG scenes.

The following code is a lightweight wrapper around SVG files. The metaphor
is to construct a scene, add objects to it, and then write it to a file
to display it.

This program uses ImageMagick to display the SVG files. ImageMagick also
does a remarkable job of converting SVG files into other formats.
"""

import os

def colorstr(rgb):
    if type(rgb) == tuple:
        return "#%02x%02x%02x" % rgb
    else:
        return rgb

def compute_style(style):
    style_str = []
    color = style.get("color", "none")
    if color is not None:
        style_str.append('fill:%s' % (colorstr(color),))
    stroke = style.get("stroke")
    if stroke is not None:
        style_str.append('stroke:%s' % (colorstr(stroke),))


    style_str = 'style="%s"' % (';'.join(style_str),)
    return style_str

class Scene:
    def __init__(self, size=(400,400)):
        self.items = []
        self.size = size

    def add(self,item):
        self.items.append(item)

    def prepend(self,item):
        self.items = [item] + self.items

    def strarray(self):
        var = [
            "<?xml version=\"1.0\"?>\n",
           "<svg height=\"%d\" width=\"%d\" >\n" % (self.size[1], self.size[0]),
           "  <defs> ",
           '    <marker id="arrow" markerWidth="10" markerHeight="10" refx="0" refy="3" orient="auto" markerUnits="strokeWidth" >',
           '      <path d="M0,0 L0,6 L9,3 z" fill="#000" />',
           "    </marker>",
           "  </defs>",
           " <g style=\"fill-opacity:1.0; stroke:black;\n",
           "  stroke-width:1;\">\n"
        ]
        for item in self.items: var += item.strarray()
        var += [" </g>\n</svg>\n"]
        return var

    def write_svg(self, file):
        file.writelines(self.strarray())

    def _repr_html_(self):
        return '\n'.join(self.strarray())

class Line:
    def __init__(self,start,end, arrow=False, **style_kwargs):
        self.start = start   #xy tuple
        self.end   = end     #xy tuple
        self.arrow = arrow
        self.style_kwargs = style_kwargs

    def strarray(self):
        style_str = compute_style(self.style_kwargs)

        maybe_arrow = 'marker-end="url(#arrow)" ' if self.arrow else ''
        return ["  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" %s %s/>\n" %\
                (self.start[0],self.start[1],self.end[0],self.end[1], maybe_arrow, style_str)]


class Circle:
    def __init__(self,center,radius, **style_kwargs):
        self.center = center
        self.radius = radius
        self.style_kwargs = style_kwargs

    def strarray(self):
        style_str = compute_style(self.style_kwargs)

        return [
            "  <circle cx=\"%d\" cy=\"%d\" r=\"%d\"\n" % (self.center[0], self.center[1], self.radius),
            "          %s />\n" % (style_str,)
        ]

class Rectangle:
    def __init__(self, origin, size, **style_kwargs):
        self.origin = origin
        self.size = size
        self.style_kwargs = style_kwargs

    def strarray(self):
        style_str = compute_style(self.style_kwargs)

        return [
            "  <rect x=\"%d\" y=\"%d\" height=\"%d\"\n" % (self.origin[0], self.origin[1], self.size[1]),
            "        width=\"%d\" %s />\n" % (self.size[0], style_str)
        ]

class Text:
    def __init__(self,origin,text,size=24):
        self.origin = origin
        self.text = text
        self.size = size
        return

    def strarray(self):
        return ["  <text x=\"%d\" y=\"%d\" font-size=\"%d\">\n" %\
                (self.origin[0],self.origin[1],self.size),
                "   %s\n" % self.text,
                "  </text>\n"]




def test():
    scene = Scene()
    scene.add(Rectangle((100,100),200,200,(0,255,255)))
    scene.add(Line((200,200),(200,300)))
    scene.add(Line((200,200),(300,200)))
    scene.add(Line((200,200),(100,200)))
    scene.add(Line((200,200),(200,100)))
    scene.add(Circle((200,200),30,(0,0,255)))
    scene.add(Circle((200,300),30,(0,255,0)))
    scene.add(Circle((300,200),30,(255,0,0)))
    scene.add(Circle((100,200),30,(255,255,0)))
    scene.add(Circle((200,100),30,(255,0,255)))
    scene.add(Text((50,50),"Testing SVG"))
    with open("test.svg", "w") as f:
        scene.write_svg(f)

if __name__ == '__main__':
    test()
