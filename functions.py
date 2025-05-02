
import matplotlib.pyplot as plt
import numpy as np
import random as rd


def is_nan(x):
    return (x is np.nan or x != x)


def rec_ave(n, xold, x, x10):
    return xold + 1 / n * (x - x10)



def mov_ave(a, n=100):
    if n > 0:
        list_mov_ave = a[0:n]
        x_old = sum(list_mov_ave) / n
        for i in range(n, len(a)):
            x_new = rec_ave(n, x_old, a[i], a[i - n])
            list_mov_ave.append(x_new)
            x_old = x_new
        return list_mov_ave
    else:
        return a


def recursive_variance(data):
    if len(data) == 1:
        return 0
    else:
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        # return math.sqrt(variance + recursive_variance(data[:-1]))
        return variance + recursive_variance(data[:-1])


def mov_var(a, n=50):
    if n > 0:
        list_var = a[0:n]
        var_old = 0
        for i in range(n, len(a)):
            var_new = recursive_variance(a[i - n:i])
            list_var.append(var_new)
            var_old = var_new
        return list_var
    else:
        return a


def plot_corr(dataframe, size=80):

    corr = dataframe.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)


def patch_spine_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def print_lengly_ave(vd, eb, bitzer, direct, dd,step, length=360):
    leng = len(eb)
    print('Average ', int(length/30), 'hourly m_dot : ', 'VD', ' ', 'EB', ' ', 'Bitzer', ' ', 'Direct', ' ', 'Data Driven')
    for i in range(1, round(leng / length * step )):
        i *= length
        ii = int(i - length)
        di = i - ii
        try:
            print(round(sum(vd[ii:i]) / di, 3), ' ',
                  round(sum(eb[ii:i]) / di, 3), ' ',
                  round(sum(bitzer[ii:i]) / di, 3), ' ',
                  round(sum(direct[ii:i]) / di, 3), ' ',
                  round(sum(dd[ii:i]) / di, 3)
                  )
        except:
            print('devision by zero')


def print_weekly_ave(vd, eb, bitzer, direct, dd):
    leng = len(eb)
    print('Average weekly m_dot: ', 'VD', ' ', 'EB', ' ', 'Bitzer', ' ', 'Direct', ' ', 'Data Driven')

    print(round(sum(vd) / leng, 3), ' ',
          round(sum(eb) / leng, 3), ' ',
          round(sum(bitzer) / leng, 3), ' ',
          round(sum(direct) / leng, 3), ' ',
          round(sum(dd) / leng, 3))
def plot_1(time, var1, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()


    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))

    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])


    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))



    host.set_xlabel(label[0])
    host.set_ylabel(label[1])

    host.yaxis.label.set_color(p1.get_color())


    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)


    host.tick_params(axis='x', **tkw)

    lines = [p1]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ' and ' +   label[2] + ' as a function of ' + label[0] + ' week ' + str(week))

def plot_2(time: object, var1: object, var2: object, label: object, week: object) -> object:

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()


    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))

    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    span_h = max([max(var1), max(var2)]) * 1.1
    span_l = min([min(var1), min(var2)]) * 0.9
    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ' and ' +   label[2] + ' as a function of ' + label[0] + ' week ' + str(week))


def plot_2_maxed(time, var1, var2, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()



    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))



    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)




    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])


    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))


    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))


    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])


    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())



    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    # plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + '  & ' + label[5] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_2_ax(time, var1, var2, label, week):

    """ 
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel 
        g : green           "s"	: square           "o" : circle 
        r : red             "p"	: pentagon         "v" : triangle_down 
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond  
        w : white           "+"	: plus                       
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()


    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))

    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    span_h = max([max(var1), max(var2)]) * 1.1
    span_l = min([min(var1), min(var2)]) * 0.9
    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ' and ' +   label[2] + ' as a function of ' + label[0] + ' week ' + str(week))


def plot_3(time, var1, var2, var3, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.08))
    par3.spines["right"].set_position(("axes", 1.16))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])


    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))
    par2.set_ylim(min(var3), max(var3))


    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])


    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + '& ' + label[3] + ' as a function of ' + label[0] + ' for week ' + str(week))




def plot_4(time, var1, var2, var3, var4, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.08))
    par3.spines["right"].set_position(("axes", 1.16))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    span_h = max([max(var1), max(var2), max(var3), max(var4)]) * 1.1
    span_l = min([min(var1), min(var2), min(var3), min(var4)]) * 0.9

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    par3.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ' & ' + label[4] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_4_maxed(time, var1, var2, var3, var4, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))

    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)


    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])


    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))


    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))
    par2.set_ylim(min(var3), max(var3))
    par3.set_ylim(min(var4), max(var4))

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())


    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    # plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + '  & ' + label[5] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))



def plot_5(time, var1, var2, var3, var4, var5, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    span_h = max([max(var1), max(var2), max(var3), max(var4), max(var5)]) * 1.1
    span_l = min([min(var1), min(var2), min(var3), min(var4), min(var5)]) * 0.9

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    par3.set_ylim(span_l, span_h)
    par4.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    # plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + '  & ' + label[5] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_5_maxed(time, var1, var2, var3, var4, var5, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))


    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))
    par2.set_ylim(min(var3), max(var3))
    par3.set_ylim(min(var4), max(var4))
    par4.set_ylim(min(var5), max(var5))

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    # plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + '  & ' + label[5] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_5_Scalable(time, var1, var2, var3, var4, var5, label,scale):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])

    try:
        host.set_xlim(min(time), max(time))
    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    # span_h = max([max(var1), max(var2), max(var3), max(var4), max(var5)]) * 1.1
    # span_l = min([min(var1), min(var2), min(var3), min(var4), min(var5)]) * 0.9


    host.set_ylim(scale[0][0], scale[0][1])
    par1.set_ylim(scale[1][0], scale[1][1])
    par2.set_ylim(scale[2][0], scale[2][1])
    par3.set_ylim(scale[3][0], scale[3][1])
    par4.set_ylim(scale[4][0], scale[4][1])

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]
    plt.grid(True)

    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + '  & ' + label[5] + ', ' + ' as a function of ' + label[0] )

    """ Plot scaling """



    # scale = [[0, 4],  # Measured Capacitance
    #          [0, 4],  # Moving average Capacitance
    #          [0, 0.8],  # Variance of capacitance
    #          [8, 25],  # Measured temperature
    #          [0, 20]]  # Variance of Tempreture


def plot_6(time, var1, var2, var3, var4, var5, var6, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    par5.spines["right"].set_position(("axes", 1.28))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))
    span_h = max([max(var1), max(var2), max(var3), max(var4), max(var5), max(var6)]) * 1.1
    span_l = min([min(var1), min(var2), min(var3), min(var4), min(var5), min(var6)]) * 0.9

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    par3.set_ylim(span_l, span_h)
    par4.set_ylim(span_l, span_h)
    par5.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5, p6]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ' & ' + label[6] + ' & ' +  ' as a function of ' + label[0] + ' for week ' + str(week))

def plot_6_maxed(time, var1, var2, var3, var4, var5, var6, label, week):

    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    par5.spines["right"].set_position(("axes", 1.28))
    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1)))*120)
        host.set_xlim(min(time), max(time))

    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))
    par2.set_ylim(min(var3), max(var3))
    par3.set_ylim(min(var4), max(var4))
    par4.set_ylim(min(var5), max(var5))
    par5.set_ylim(min(var6), max(var6))

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5, p6]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ' & ' + label[6] + ' & ' +  ' as a function of ' + label[0] + ' for week ' + str(week))



def plot_7(time, var1, var2, var3, var4, var5, var6, var7, label, week):
    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    par5.spines["right"].set_position(("axes", 1.28))
    par6.spines["right"].set_position(("axes", 1.35))

    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)
    par6.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])
    p7, = par6.plot(time, var7, "k-", label=label[7])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1))) * 120)
        host.set_xlim(min(time), max(time))

    span_h = max([max(var1), max(var2), max(var3), max(var4), max(var5), max(var6), max(var7)]) * 1.05
    span_l = min([min(var1), min(var2), min(var3), min(var4), min(var5), min(var6), min(var7)]) * 0.95

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    par3.set_ylim(span_l, span_h)
    par4.set_ylim(span_l, span_h)
    par5.set_ylim(span_l, span_h)
    par6.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])
    par6.set_ylabel(label[7])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())
    par6.yaxis.label.set_color(p7.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
    par6.tick_params(axis='y', colors=p7.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3, p4, p5, p6, p7]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ', ' + label[6] + ' & ' + label[7] + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_7_maxed(time, var1, var2, var3, var4, var5, var6, var7, label, week):
    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.06))
    par3.spines["right"].set_position(("axes", 1.12))
    par4.spines["right"].set_position(("axes", 1.18))
    par5.spines["right"].set_position(("axes", 1.24))
    par6.spines["right"].set_position(("axes", 1.30))

    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)
    par6.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])
    p7, = par6.plot(time, var7, "k-", label=label[7])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1))) * 120)
        host.set_xlim(min(time), max(time))

    host.set_ylim(min(var1), max(var1))
    par1.set_ylim(min(var2), max(var2))
    par2.set_ylim(min(var3), max(var3))
    par3.set_ylim(min(var4), max(var4))
    par4.set_ylim(min(var5), max(var5))
    par5.set_ylim(min(var6), max(var6))
    par6.set_ylim(min(var7), max(var7))

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])
    par6.set_ylabel(label[7])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())
    par6.yaxis.label.set_color(p7.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
    par6.tick_params(axis='y', colors=p7.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3, p4, p5, p6, p7]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ', ' + label[6] + ' & ' + label[7] + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_8(time, var1, var2, var3, var4, var5, var6, var7,var8, label, week):
    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.canvas.set_window_title(str(week))
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()
    par7 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    par5.spines["right"].set_position(("axes", 1.28))
    par6.spines["right"].set_position(("axes", 1.35))
    par7.spines["right"].set_position(("axes", 1.42))

    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)
    par6.spines["right"].set_visible(True)
    par7.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "b-", label=label[1])
    p2, = par1.plot(time, var2, "g-", label=label[2])
    p3, = par2.plot(time, var3, "r-", label=label[3])
    p4, = par3.plot(time, var4, "c-", label=label[4])
    p5, = par4.plot(time, var5, "m-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])
    p7, = par6.plot(time, var7, "k-", label=label[7])
    p8, = par7.plot(time, var8, "b-", label=label[8])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1))) * 120)
        host.set_xlim(min(time), max(time))

    span_h = max([max(var1), max(var2), max(var3), max(var4), max(var5), max(var6), max(var7), max(var8)]) * 1.05
    span_l = min([min(var1), min(var2), min(var3), min(var4), min(var5), min(var6), min(var7), min(var8)]) * 0.95

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    par3.set_ylim(span_l, span_h)
    par4.set_ylim(span_l, span_h)
    par5.set_ylim(span_l, span_h)
    par6.set_ylim(span_l, span_h)
    par7.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])
    par6.set_ylabel(label[7])
    par7.set_ylabel(label[8])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())
    par6.yaxis.label.set_color(p7.get_color())
    par7.yaxis.label.set_color(p8.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
    par6.tick_params(axis='y', colors=p7.get_color(), **tkw)
    par7.tick_params(axis='y', colors=p8.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3, p4, p5, p6, p7,p8]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ', ' + label[6] + ', ' + label[7] + ' & ' + label[8] + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_4_ej(time, var1, var2, var3, var4, label, week):
    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))


    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])


    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1))) * 120)
        host.set_xlim(min(time), max(time))

    span_h = max(var1) * 1.02
    span_l = min(var1) * 0.98
    host.set_ylim(span_l, span_h)


    span_h = max([max(var2), max(var3)]) * 1.02
    span_l = min([min(var2), min(var3)]) * 0.98
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)

    span_h = max(var4) * 1.02
    span_l = min(var4) * 0.98
    par3.set_ylim(span_l, span_h)




    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)

    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3, p4]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + '& ' + label[4] + ', ' + ' as a function of ' + label[0] + ' for week ' + str(week))


def plot_7_ej(time, var1, var2, var3, var4, var5, var6, var7, label, week):
    """
    First plots the lines and then 'ro' spesifies red dots as the TDN states __ r ::  for red and o :: for circle
        Color               Shape                  shape
        b : blue            "8"	: octagon          "," : pixel
        g : green           "s"	: square           "o" : circle
        r : red             "p"	: pentagon         "v" : triangle_down
        c : cyan            "P"	: plus (filled)    "x" : x
        m : magenta         "*"	: star             "X" : x (filled)
        y : yellow          "h"	: hexagon1         "D" : diamond
        k : black           "H"	: hexagon2         "d" : thin_diamond
        w : white           "+"	: plus
    """

    fig, host = plt.subplots()

    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()

    """Offset the right spine of par2.  The ticks and label have already been
     placed on the right by twinx above."""
    par1.spines["right"].set_position(("axes", 1))
    par2.spines["right"].set_position(("axes", 1.07))
    par3.spines["right"].set_position(("axes", 1.14))
    par4.spines["right"].set_position(("axes", 1.21))
    par5.spines["right"].set_position(("axes", 1.28))
    par6.spines["right"].set_position(("axes", 1.35))

    patch_spine_invisible(par2)
    """Second, show the right spine."""
    par1.spines["right"].set_visible(True)
    par2.spines["right"].set_visible(True)
    par3.spines["right"].set_visible(True)
    par4.spines["right"].set_visible(True)
    par5.spines["right"].set_visible(True)
    par6.spines["right"].set_visible(True)

    p1, = host.plot(time, var1, "c-", label=label[1])
    p2, = par1.plot(time, var2, "r-", label=label[2])
    p3, = par2.plot(time, var3, "m-", label=label[3])
    p4, = par3.plot(time, var4, "b-", label=label[4])
    p5, = par4.plot(time, var5, "g-", label=label[5])
    p6, = par5.plot(time, var6, "y-", label=label[6])
    p7, = par5.plot(time, var7, "k-", label=label[7])

    try:
        host.set_xlim(min(time), max(time))

    except:
        time = []
        time = list(np.array(range(len(var1))) * 120)
        host.set_xlim(min(time), max(time))

    span_h = max([max(var1), max(var2), max(var3)]) * 1.02
    span_l = min([min(var1), min(var2), min(var3)]) * 0.98

    host.set_ylim(span_l, span_h)
    par1.set_ylim(span_l, span_h)
    par2.set_ylim(span_l, span_h)
    span_h = max([max(var4), max(var5)]) * 1.02
    span_l = min([min(var4), min(var5)]) * 0.98
    par3.set_ylim(span_l, span_h)
    par4.set_ylim(span_l, span_h)

    span_h = max([max(var6), max(var7)]) * 1.02
    span_l = min([min(var6), min(var7)]) * 0.98
    par5.set_ylim(span_l, span_h)
    par6.set_ylim(span_l, span_h)

    host.set_xlabel(label[0])
    host.set_ylabel(label[1])
    par1.set_ylabel(label[2])
    par2.set_ylabel(label[3])
    par3.set_ylabel(label[4])
    par4.set_ylabel(label[5])
    par5.set_ylabel(label[6])
    par6.set_ylabel(label[7])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())
    par4.yaxis.label.set_color(p5.get_color())
    par5.yaxis.label.set_color(p6.get_color())
    par6.yaxis.label.set_color(p7.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    par4.tick_params(axis='y', colors=p5.get_color(), **tkw)
    par5.tick_params(axis='y', colors=p6.get_color(), **tkw)
    par6.tick_params(axis='y', colors=p7.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3, p4, p5, p6, p7]
    plt.grid(True)
    host.legend(lines, [l.get_label() for l in lines])
    plt.title(label[1] + ', ' + label[2] + ', ' + label[3] + ', ' + label[4] + ', ' + label[5] + ', ' + label[6] + ' & ' + label[7] + ' as a function of ' + label[0] + ' for week ' + str(week))


def sct_fit(x, y, po, col, label):
    plt.scatter(x, y, marker='.', color=col, s=1)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, po))(np.unique(x)), color=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)


def sct_ave_groupby(df, po, col, label):
    plt.scatter(df[label[1]],df[label[2]], marker='.', color=col, s=1)
    # print(np.polyfit(df[label[1]], df[label[2]], po))
    plt.plot(np.unique(df[label[1]]), np.poly1d(np.polyfit(df[label[1]], df[label[2]], po))(np.unique(df[label[1]])), color=col)
    df = df.groupby(label[1]).mean()
    plt.scatter(df.index, df[label[2]], c=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)


def sct_df(df, po, col, label, n):
    x=df[label[1]].tolist()
    y=df[label[2]].tolist()
    plt.scatter(df[label[1]], df[label[2]], marker='.', color=col, s=1)
    # print(len(mov_ave(df[label[2]], 100)))
    plt.scatter(df[label[1]], mov_ave(df[label[2]].tolist(), n), marker=',', color=(rd.random(), rd.random(), rd.random()), s=1)
    # plt.scatter(x, np.array(y) - np.array(mov_ave(y, 100)), marker=',',color=(rd.random(), rd.random(), rd.random()), s=1)
    y_diff = np.array(y) - np.array(mov_ave(y, n)) - abs(np.array(y) - np.array(mov_ave(y, n)))
    # plt.scatter(y,y_diff)

    # y=y
    y=y_diff
    dx = x[1] - x[0]
    dydx = np.gradient(y, dx)
    plt.scatter(x, dydx, marker=',', color=(rd.random(), rd.random(), rd.random()),s=1)

    # print(np.polyfit(df[label[1]], df[label[2]], po))
    # plt.plot(np.unique(df[label[1]]), np.poly1d(np.polyfit(df[label[1]], df[label[2]], po))(np.unique(df[label[1]])),color=col)
    # plt.scatter(df.index, df[label[2]], c=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)

def sct(x, y, po, col, label):
    plt.scatter(x, y, marker='.', color=col, s=1)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, po))(np.unique(x)), color=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)


def sctFitAve(x, y, po, col, label):
    plt.scatter(x, y, marker='.', color=col, s=1)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, po))(np.unique(x)), color=col)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 0))(np.unique(x)), '--', color=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)


def sctMVAve(x, y, n, col, label):

    plt.scatter(x, mov_ave(y, n), marker='.', color=col, s=1)
    plt.plot(x, [0]*len(x), marker='.', color=col)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 0))(np.unique(x)), '--', color=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)


def fit(x, y, po, col, label):
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, po))(np.unique(x)), color=col)
    plt.xlabel(label[1])
    plt.ylabel(label[2])
    plt.grid(c='grey', linestyle='-', linewidth=0.5, alpha=0.7)
