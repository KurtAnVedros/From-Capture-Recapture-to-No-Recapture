#!/usr/bin/env python
# coding: utf-8

# # Information
# - notebook contains all basic information of each executable assembly instruction for a ATmega 2560 device.
# - All information obtained from https://ww1.microchip.com/downloads/en/devicedoc/atmel-0856-avr-instruction-set-manual.pdf 

# In[3]:


import numpy as np
import timeit
import math 
import random as rand


# In[14]:


listCodes = []
    
## Information on the possible codes
## organized as {name, cycles, colorOnGraph, IfUsedOnce}
listCodes.append({"Name":"add", "Cycles":1, "Color":"red"})
listCodes.append({"Name":"adc", "Cycles":1, "Color":"red"})
listCodes.append({"Name":"adiw", "Cycles":2, "Color":"silver"})
listCodes.append({"Name":"and", "Cycles":1, "Color":"tab:Brown"})
listCodes.append({"Name":"andi", "Cycles":1, "Color":"tab:Brown"})
listCodes.append({"Name":"asr", "Cycles":1, "Color":"rosybrown"})
listCodes.append({"Name":"bclr", "Cycles":1, "Color":"lightcoral"})
listCodes.append({"Name":"bld", "Cycles":1, "Color":"indianred"})
listCodes.append({"Name":"brbs-f", "Cycles":1, "Color":"brown"})
listCodes.append({"Name":"brbs-t", "Cycles":2, "Color":"bisque"})
listCodes.append({"Name":"brcc-f", "Cycles":1, "Color":"darkorange"})
listCodes.append({"Name":"brcc-t", "Cycles":2, "Color":"burlywood"})
listCodes.append({"Name":"brcs-f", "Cycles":1, "Color":"darkorange"})
listCodes.append({"Name":"brcs-t", "Cycles":2, "Color":"burlywood"})
listCodes.append({"Name":"break", "Cycles":1, "Color":"tan"})
listCodes.append({"Name":"breq-f", "Cycles":1, "Color":"peru"})
listCodes.append({"Name":"breq-t", "Cycles":2, "Color":"firebrick"})
listCodes.append({"Name":"brge-f", "Cycles":1, "Color":"olive"})
listCodes.append({"Name":"brge-t", "Cycles":2, "Color":"grey"})
listCodes.append({"Name":"brlo-f", "Cycles":1, "Color":"navajowhite"})
listCodes.append({"Name":"brlo-t", "Cycles":2, "Color":"orange"})
listCodes.append({"Name":"brmi-f", "Cycles":1, "Color":"navajowhite"})
listCodes.append({"Name":"brmi-t", "Cycles":2, "Color":"orange"})
listCodes.append({"Name":"brne-f", "Cycles":1, "Color":"magenta"}) 
listCodes.append({"Name":"brne-t", "Cycles":2, "Color":"orchid"})
listCodes.append({"Name":"brpl-f", "Cycles":1, "Color":"gold"})
listCodes.append({"Name":"brpl-t", "Cycles":2, "Color":"goldenrod"})
listCodes.append({"Name":"brsh-f", "Cycles":1, "Color":"darkgoldenrod"})
listCodes.append({"Name":"brsh-t", "Cycles":2, "Color":"khaki"})
listCodes.append({"Name":"bset", "Cycles":1, "Color":"darkkhaki"})
listCodes.append({"Name":"bst", "Cycles":1, "Color":"forestgreen"})
listCodes.append({"Name":"call", "Cycles":5, "Color":"aqua"})
listCodes.append({"Name":"cbi", "Cycles":2, "Color":"limegreen"})
listCodes.append({"Name":"cbr", "Cycles":1, "Color":"lime"})
listCodes.append({"Name":"clc", "Cycles":1, "Color":"aquamarine"})
listCodes.append({"Name":"clh", "Cycles":1, "Color":"tab:olive"})
listCodes.append({"Name":"cli", "Cycles":1, "Color":"lightseagreen"})
listCodes.append({"Name":"cln", "Cycles":1, "Color":"darkorchid"})
listCodes.append({"Name":"clr", "Cycles":1, "Color":"salmon"})
listCodes.append({"Name":"cls", "Cycles":1, "Color":"dimgrey"})
listCodes.append({"Name":"clt", "Cycles":1, "Color":"deeppink"})
listCodes.append({"Name":"clv", "Cycles":1, "Color":"fuchsia"})
listCodes.append({"Name":"clz", "Cycles":1, "Color":"salmon"})
listCodes.append({"Name":"com", "Cycles":1, "Color":"royalblue"})
listCodes.append({"Name":"cp", "Cycles":1, "Color":"y"})
listCodes.append({"Name":"cpi", "Cycles":1, "Color":"y"})
listCodes.append({"Name":"cpse-f", "Cycles":1, "Color":"mediumaquamarine"})
listCodes.append({"Name":"cpse-t-1", "Cycles":2, "Color":"darkslategray"})
listCodes.append({"Name":"cpse-t-2", "Cycles":3, "Color":"darkslategrey"})
listCodes.append({"Name":"dec", "Cycles":1, "Color":"dimgray"})
listCodes.append({"Name":"elpm", "Cycles":3, "Color":"teal"})
listCodes.append({"Name":"eor", "Cycles":1, "Color":"purple"})
listCodes.append({"Name":"in", "Cycles":1, "Color":"darkcyan"})
listCodes.append({"Name":"inc", "Cycles":1, "Color":"mediumvioletred"})
listCodes.append({"Name":"ld", "Cycles":2, "Color":"slategrey"})
listCodes.append({"Name":"ld+", "Cycles":2, "Color":"slategrey"})
listCodes.append({"Name":"ld-", "Cycles":2, "Color":"cornflowerblue"})
listCodes.append({"Name":"ldd", "Cycles":2, "Color":"royalblue"})
listCodes.append({"Name":"ldi", "Cycles":1, "Color":"darkorange"})
listCodes.append({"Name":"lds", "Cycles":2, "Color":"navy"})
listCodes.append({"Name":"lpm", "Cycles":3, "Color":"midnightblue"})
listCodes.append({"Name":"lsl", "Cycles":1, "Color":"turquoise"})
listCodes.append({"Name":"lsr", "Cycles":1, "Color":"darkviolet"})
listCodes.append({"Name":"mov", "Cycles":1, "Color":"springgreen"})
listCodes.append({"Name":"mul", "Cycles":2, "Color":"darkgray"})
listCodes.append({"Name":"muls", "Cycles":2, "Color":"darkgray"})
listCodes.append({"Name":"fmul", "Cycles":2, "Color":"purple"})
listCodes.append({"Name":"neg", "Cycles":1 , "Color":"darkmagenta"})
listCodes.append({"Name":"nop", "Cycles":1, "Color":"green"})
listCodes.append({"Name":"or", "Cycles":1, "Color":"black"})
listCodes.append({"Name":"ori", "Cycles":1, "Color":"black"})
listCodes.append({"Name":"out", "Cycles":1, "Color":"tab:red"})
listCodes.append({"Name":"pop", "Cycles":2, "Color":"yellow"})
listCodes.append({"Name":"push", "Cycles":2, "Color":"rebeccapurple"})
listCodes.append({"Name":"jmp", "Cycles":3, "Color":"cyan"})
listCodes.append({"Name":"rcall", "Cycles":4, "Color":"blueviolet"})
listCodes.append({"Name":"ret", "Cycles":5, "Color":"indigo"})
listCodes.append({"Name":"rol", "Cycles":1, "Color":"darkorchid"})
listCodes.append({"Name":"ror", "Cycles":1, "Color":"darkviolet"})
listCodes.append({"Name":"rjmp", "Cycles":2, "Color":"steelblue"})
listCodes.append({"Name":"sbc", "Cycles":1, "Color":"crimson"})
listCodes.append({"Name":"sbci", "Cycles":1, "Color":"darkgreen"})
listCodes.append({"Name":"sbi", "Cycles":2, "Color":"orange"})
listCodes.append({"Name":"sbiw", "Cycles":2, "Color":"violet"})
listCodes.append({"Name":"sbis-f", "Cycles":1, "Color":"deeppink"})
listCodes.append({"Name":"sbis-t-1", "Cycles":2, "Color":"crimson"})
listCodes.append({"Name":"sbis-t-2", "Cycles":3, "Color":"dodgerblue"})
listCodes.append({"Name":"sbr", "Cycles":1, "Color":"purple"})
listCodes.append({"Name":"sbrc-f", "Cycles":1, "Color":"deeppink"})
listCodes.append({"Name":"sbrc-t-1", "Cycles":2, "Color":"crimson"})
listCodes.append({"Name":"sbrc-t-2", "Cycles":3, "Color":"dodgerblue"})
listCodes.append({"Name":"sec", "Cycles":1, "Color":"peru"})
listCodes.append({"Name":"sen", "Cycles":1, "Color":"tomato"})
listCodes.append({"Name":"ser", "Cycles":1, "Color":"orange"})
listCodes.append({"Name":"ses", "Cycles":1, "Color":"deepskyblue"})
listCodes.append({"Name":"sev", "Cycles":1, "Color":"olivedrab"})
listCodes.append({"Name":"sez", "Cycles":1, "Color":"magenta"})
listCodes.append({"Name":"sleep", "Cycles":1, "Color":"saddlebrown"})
listCodes.append({"Name":"spm", "Cycles":4, "Color":"orangered"})
listCodes.append({"Name":"st", "Cycles":2, "Color":"tomato"})
listCodes.append({"Name":"st+", "Cycles":2, "Color":"tomato"})
listCodes.append({"Name":"st-", "Cycles":2, "Color":"greenyellow"})
listCodes.append({"Name":"std", "Cycles":2, "Color":"cadetblue"})
listCodes.append({"Name":"sts", "Cycles":2, "Color":"darkcyan"})
listCodes.append({"Name":"sub", "Cycles":1, "Color":"lawngreen"})
listCodes.append({"Name":"swap", "Cycles":1, "Color":"steelblue"})
listCodes.append({"Name":"tst", "Cycles":1, "Color":"blue"})
listCodes.append({"Name":"wdr", "Cycles":1, "Color":"deepskyblue"})


# In[8]:


len(listCodes)


# In[10]:


def getCycles(name):
    for i in range(len(listCodes)):
        if listCodes[i]["Name"] == name:
            return listCodes[i]["Cycles"]
        
    return "No instruction found with that name. Make sure in lower case. If a branch be sure to indicate via t or f at the end for true or false. Example: brne-t"


# In[9]:


def getColor(name):
    for i in range(len(listCodes)):
        if listCodes[i]["Name"] == name:
            return listCodes[i]["Color"]
        
    return "No instruction found with that name. Make sure in lower case. If a branch be sure to indicate via t or f at the end for true or false. Example: brne-t"


# In[15]:


getColor("rcall")

