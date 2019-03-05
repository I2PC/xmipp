## Frame motion param estimator
## Made in Octave 4.2.2

close all
clear h

graphics_toolkit qt

h.ax = axes ("position", [0.05 0.42 0.9 0.5]);

function update_plot (obj, init = false)

  ## gcbo holds the handle of the control
  h = guidata (obj);
  replot = false;
  recalc = false;
  switch (gcbo)
    case {h.a1}
      recalc = true;
    case {h.a2}
      recalc = true;
    case {h.b1}
      recalc = true;
    case {h.b2}
      recalc = true;
  endswitch
  if (recalc || init)
    a1 = (get (h.a1, "value") - 0.5) / 10;
    a2 = (get (h.a2, "value") - 0.5) / 10;
    b1 = (get (h.b1, "value") - 0.5) / 10;
    b2 = (get (h.b2, "value") - 0.5) / 10;
    set (h.a1_label, "string", sprintf ("a1: %.3f", a1));    
    set (h.a2_label, "string", sprintf ("a2: %.3f", a2));
    set (h.b1_label, "string", sprintf ("b1: %.3f", b1));
    set (h.b2_label, "string", sprintf ("b2: %.3f", b2));
    t=0:1:60;
    x = a1.*t + a2.*t.*t + (cos(t/10))/10
    y = b1.*t + b2.*t.*t + (sin(t.*t))/5
    if (init)
      h.plot = plot (x, y);
      guidata (obj, h);
    else
      set (h.plot, "ydata", y);
      set (h.plot, "xdata", x);
    endif
  endif
  
endfunction

# controls
h.a1_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "a1:",
                           "horizontalalignment", "left",
                           "position", [0.05 0.3 0.35 0.08]);

h.a1 = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "slider",
                            "callback", @update_plot,
                            "value", 0.5,
                            "position", [0.05 0.25 0.35 0.06]);
                            
h.a2_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "a2:",
                           "horizontalalignment", "left",
                           "position", [0.05 0.15 0.35 0.08]);

h.a2 = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "slider",
                            "callback", @update_plot,
                            "value", 0.5,
                            "position", [0.05 0.10 0.35 0.06]);
                            
h.b1_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "b1:",
                           "horizontalalignment", "left",
                           "position", [0.5 0.3 0.35 0.08]);

h.b1 = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "slider",
                            "callback", @update_plot,
                            "value", 0.5,
                            "position", [0.5 0.25 0.35 0.06]);
                            
h.b2_label = uicontrol ("style", "text",
                           "units", "normalized",
                           "string", "b2:",
                           "horizontalalignment", "left",
                           "position", [0.5 0.15 0.35 0.08]);

h.b2 = uicontrol ("style", "slider",
                            "units", "normalized",
                            "string", "slider",
                            "callback", @update_plot,
                            "value", 0.5,
                            "position", [0.5 0.10 0.35 0.06]);




set (gcf, "color", get(0, "defaultuicontrolbackgroundcolor"))
guidata (gcf, h)
update_plot (gcf, true);
