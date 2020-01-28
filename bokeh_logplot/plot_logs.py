# -*- coding: utf-8 -*-
"""
============================================================
Log plotting
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      January 2020
 License:   GPL v3


@author: paul
"""

import pandas as pd
from bokeh.plotting import figure
from bokeh.palettes import all_palettes
from bokeh.models.tickers import FixedTicker
from bokeh.models import LinearAxis, Range1d, LogAxis
from bokeh.layouts import gridplot
from collections import OrderedDict

scaledlog = pd.DataFrame()

def removeNone(lst):
    return [x for x in lst if x != 'None']

def getShadingBounds(inputlog):
    minbound = inputlog.min()
    maxbound = inputlog.max()
    extension = (maxbound - minbound) / 20
    minbound = minbound - extension
    maxbound =  maxbound + extension
    return(minbound, maxbound)

def getScaledLog(nr, minval, maxval, minvalues, maxvalues, data, logname):
    if (nr <= 1) :
        scalingfactor = 1
        scalingintercept = 0
    else:
        targetrange = float(maxvalues[0]) - float(minvalues[0])
        logrange = float(maxval) - float(minval)
        scalingfactor = targetrange / logrange
        scalingintercept = (-targetrange * float(minval) / logrange 
                            + float(minvalues[0]))
        minvalues[nr] = minvalues[0]
        maxvalues[nr] = maxvalues[0]
    scaledlog = (data.loc[:, logname].copy() * scalingfactor + scalingintercept)
    return (scaledlog, minvalues, maxvalues)
    
def create_gridplot(alldata):
    nrlogplots = int(alldata['nrlogplots'].value)
    a = {}
    plotlist = []
    for nr in range(nrlogplots):
        key = 'plot'+str(nr)
        a[key] = create_plot(alldata, nr)
        plotlist.append(a[key])
        if (key == 'plot0'):
            y_range = a[key].y_range
        else:
            a[key].y_range = y_range
    grid = gridplot([plotlist])
    return(grid)

def add_shading(plot, scaledlogs, depthdata, lognames, xrange2,
                shadingtype, shadinglogs, linestyles, mypalette,
                colorfillpalette, colorfill, shadingcolor):
    leftbound = pd.DataFrame()
    rightbound = pd.DataFrame()
    shadingdifferencelog1 = 'None'
    shadingdifferencelog2 = 'None'
    firstcurve = scaledlogs.loc[:, shadinglogs[0]]
    (minbound, maxbound) = getShadingBounds(firstcurve)
    leftbound = pd.DataFrame()
    rightbound = pd.DataFrame()
    shadingdifferencelog1 = 'None'
    shadingdifferencelog2 = 'None'
    if (shadingtype == 'Difference 2 logs shading color'):
        if (len(shadinglogs) >= 2):
            shadingdifferencelog1 = shadinglogs[0]
            shadingdifferencelog2 = shadinglogs[1]
            secondcurve = scaledlogs.loc[:, shadinglogs[1]]
            print ("Warning: a difference plot cannot be combined with",
                   "curve plots. Selected curves are ignored.",
                   "Similarly, the shading is applied between the",
                   "first and second log; additional selections are ignored.")
        else:
            print('error: select 2 logs for difference shading')
    if ((shadingtype == 'Left palette') or
            (shadingtype == 'Left shading color')):
        leftbound = pd.Series(minbound for x in range(len(firstcurve)))
        rightbound = firstcurve
    if ((shadingtype == 'Right palette') or
            (shadingtype == 'Right shading color')):
        leftbound = firstcurve
        rightbound = pd.Series(maxbound for x in range(len(firstcurve)))
    if (shadingtype == 'Column wide palette'):
        leftbound = pd.Series(minbound for x in range(len(firstcurve)))
        rightbound = pd.Series(maxbound for x in range(len(firstcurve)))
    if (shadingtype == 'Difference 2 logs shading color'):
        leftbound = firstcurve
        rightbound = secondcurve
        plot.line(leftbound, depthdata[:],
                  legend_label=shadingdifferencelog1,
                  color=mypalette[-2],
                   line_width=2,
                  line_dash =  linestyles[-2])
        plot.line(rightbound, depthdata[:],
                  legend_label=shadingdifferencelog2,
                  color=mypalette[-1],
                  line_width=2,
                  line_dash =  linestyles[-1])
    minvalue = firstcurve.min()
    maxvalue = firstcurve.max()
    scalefactor = (float(255)/(float(maxvalue)-float(minvalue)))
    color = [None for _ in range(len(firstcurve))]
    if (colorfill == 'palette'):
         for x in range(len(firstcurve)):
            value = firstcurve[x] - minvalue
            color[x] = all_palettes[
                    colorfillpalette][256][int(value * scalefactor)]
    else:
         for x in range(len(firstcurve)):
             color[x] = shadingcolor

    if ((len(lognames) >= 2) and
                (shadinglogs[0] == lognames[1])):
        for x in range(len(firstcurve)):
            plot.quad(top=depthdata[x],
                      bottom=depthdata[x],
                      left=leftbound[x],
                      right=rightbound[x],
                      line_width = 1,
                      x_range_name = xrange2,
                      fill_color = color[x],
                      fill_alpha = 0.2,
                      line_color = color[x],
                      line_alpha = 0.2
                      )
    else:
        for x in range(len(firstcurve)):
            plot.quad(top=depthdata[x],
                      bottom=depthdata[x],
                      left=leftbound[x],
                      right=rightbound[x],
                      line_width = 1,
                      fill_color = color[x],
                      fill_alpha = 0.2,
                      line_color = color[x],
                      line_alpha = 0.2
                      )
    return(plot)

#Make a Bokeh plot
def create_plot(alldata, nr):
    data = alldata['data']
    xlogscales = alldata['xlogscales']
    nrcurves = alldata['nrcurves']
    xaxistype = (alldata['rowofcolcurves'].children[nr].
                    children[0].active)
    selmullogs = (alldata['rowofcolcurves'].children[nr].
                    children[1])
    shadingtype = (alldata['rowofcolcurves'].children[nr].
                    children[2].value)
    shadinglogs = list((alldata['rowofcolcurves'].children[nr].
                    children[3]).value)
    shadingcolor =  alldata['shadingcolor'].value
    colorfillpalette = alldata['colorfillpalette'].value
    colorfill = 'single'
    xrange2 = 'dummy'
    if (((shadingtype == 'Left palette') or
         (shadingtype == 'Right palette') or
         (shadingtype == 'Column wide palette'))):
        colorfill = 'palette'
    shadingdifferencelog1 = 'None'
    shadingdifferencelog2 = 'None'
    if (shadingtype == 'Difference 2 logs shading color'):
        if (len(shadinglogs) >= 2):
            shadingdifferencelog1 = shadinglogs[0]
            shadingdifferencelog2 = shadinglogs[1]
        else:
            print('error: select 2 logs for difference shading')
    depthlogstr = alldata['seldepthlog'].value
    depthdata = -data.loc[:, depthlogstr] 
    logwidth = int(alldata['logwidth'].value)
    logheight = int(alldata['logheight'].value)
    mindepth = alldata['mindepth'].value
    maxdepth = alldata['maxdepth'].value
    if (mindepth == ""):
        mindepth = depthdata.iloc[-1]
    else:
        mindepth = -float(alldata['maxdepth'].value) # a bit weird: depth input is positive but plot is negative
    if (maxdepth == ""):
        maxdepth = depthdata.iloc[0]
    else:
        maxdepth = -float(alldata['mindepth'].value)
    depthticks = int(alldata['depthticks'].value)
    rowofcol = alldata['rowofcol']
    depthminorticks = int(alldata['depthminorticks'].value)
    ylabel = depthlogstr
    lognames = list(selmullogs.value) 
    if ((shadingtype != 'None') and
            (shadinglogs[0] != 'None')):
        lognames.append(shadinglogs[0])
    if (shadingtype == 'Difference 2 logs shading color'):
        lognames = shadinglogs # No other logs allowed for difference shading
    lognames = removeNone(lognames)
    lognames = list(OrderedDict.fromkeys(lognames)) # remove double entries
    xaxistype = xlogscales[0]
    mypalette = []
    scaledlogs = pd.DataFrame()
    counter = []
    minvalues = []
    maxvalues = []
    linestyles = []
    for i in range(len(lognames)):
        for nm in range(nrcurves):
            if (lognames[i] == (rowofcol.children[0].
                                 children[nm].value)):
                minval = (rowofcol.children[1].
                                 children[nm].value)
                maxval = (rowofcol.children[2].
                                 children[nm].value)
                minvalues.append(minval)
                maxvalues.append(maxval)
                (scaledlog, minvalues, maxvalues) = getScaledLog(i, 
                                        minval, maxval,
                                        minvalues, maxvalues,
                                        data, lognames[i])
                scaledlogs = pd.concat([scaledlogs, scaledlog], axis=1)
                linestyles.append(rowofcol.children[3].
                                 children[nm].value)
                mypalette.append((rowofcol.children[4].
                                 children[nm].value))
                break
        counter.append(i)
    
    plot = figure(plot_width=logwidth, plot_height=logheight,
                 x_axis_type = xaxistype,
                 x_axis_label = lognames[0],
                 x_axis_location = 'above',
                 background_fill_color="#f0f0f0",
                 tools='pan,wheel_zoom,box_select,reset,hover,save',
                 y_axis_label=ylabel)

    ticker = []
    for i in range(0,-10000,-depthticks):
        ticker.append(i)
    minorticks = []
    for i in range(0,-10000,-depthminorticks):
        minorticks.append(i)
    plot.yaxis.ticker = FixedTicker(ticks=ticker, 
                                    minor_ticks = minorticks)
    plot.ygrid.grid_line_color = 'navy'
    plot.ygrid.grid_line_alpha = 0.2
    plot.ygrid.minor_grid_line_color = 'navy'
    plot.ygrid.minor_grid_line_alpha = 0.1
    plot.title.text = 'Plot ' + str(nr+1)
    plot.y_range = Range1d(mindepth , maxdepth) 
    plot.x_range = Range1d(float(minvalues[0]), float(maxvalues[0]))
    
    for count, name, minvalue, maxvalue, linestyle, color in zip(counter,
                                  lognames, 
                                  minvalues, 
                                  maxvalues, 
                                  linestyles,
                                  mypalette):
        if (count == 0):
            if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):
                plot.line(scaledlogs[name], depthdata[:],
                          legend_label=name,
                          color=color, 
                          line_width=2,
                          line_dash = linestyle)
        if (count == 1):
            xrange2 = name
            plot.extra_x_ranges={xrange2: Range1d(float(minvalues[1]), 
                                               float(maxvalues[1]))}
            if (shadingtype == 'Difference 2 logs shading color'):
                plot.extra_x_ranges={xrange2: Range1d(float(minvalues[0]),
                                                   float(maxvalues[0]))}
            if (xaxistype == 'linear'):           
                 plot.add_layout(LinearAxis(x_range_name=xrange2, 
                        axis_label=name), 'above')
            if (xaxistype == 'log'):
                  plot.add_layout(LogAxis(x_range_name=xrange2, 
                        axis_label=name), 'above')
            if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):
                plot.line(scaledlogs[name], depthdata[:],
                          legend_label=name,
                          color=color, 
                          x_range_name = name,
                          line_width = 2,
                          line_dash =  linestyle)  
        if (count > 1):
            if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):
                plot.line(scaledlogs[name], depthdata[:],
                          legend_label=name,
                          color=color, 
                          line_width=2,
                          line_dash =  linestyle) 
                print ("Warning: only two separate X-ranges supported.",
                       "Additional curves are scaled to the range of the first curve.")

# shading plots
    if (shadingtype != 'None'):
        plot = add_shading(plot, scaledlogs, depthdata, lognames, xrange2,
                       shadingtype, shadinglogs, linestyles, mypalette,
                       colorfillpalette, colorfill, shadingcolor)
    return (plot)