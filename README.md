AetherMMS v1.0
====================================================

Purpose
-------

AetherMMS is an open-source Python tool for GNSS mission planning of mobile
mapping surveys in urban environments. It estimates satellite visibility along
a vehicle trajectory by combining GNSS almanac propagation with terrain
elevation, building footprints, tree crowns and tunnel/covered-road segments,
and ranks candidate acquisition windows according to the predicted GNSS quality.

The tool is intentionally organised as a small set of macro-scripts, with no
package initialisation files and no hidden framework layer. All user
configuration is stored in a plain text file.


Directory structure
-------------------

    AetherMMS
    |
    |-- source
    |   |-- runAetherMMS.py
    |   |-- buildingcity.py
    |   |-- satellitepropagation.py
    |   |-- bestwindowsprediction.py
    |   |-- plotting.py
    |
    |-- config
    |   |-- config.txt
    |
    |-- Data
    |   |-- DTM5x5.tif
    |   |-- Traj.kml
    |   |-- Buildings.geojson
    |   |-- Trees.geojson
    |   |-- GNSSBases.kml
    |   |-- GPS_yuma.alm
    |   |-- Galileo.xml
    |   |-- osm_tunnels_cache.geojson
    |
    |-- results
    |   |-- csv
    |   |-- png
    |   |-- AetherMMS_report.html
    |
    |-- webUI
        |-- AetherMMS_v1.html

The example DTM is a GeoTIFF excerpt (5 m resolution, deflate-compressed)
cropped around the demo trajectory corridor; the OSM tunnel cache is included
so the demo run is fully reproducible without network access.


Execution
---------

Download or clone the repository. 
Install the Python dependencies:

    pip install -r requirements.txt

Run from the project folder:

    python source/runAetherMMS.py

Run from any other folder:

    python C:\path\to\AetherMMS\source\runAetherMMS.py

Relative paths are resolved from the AetherMMS project folder, not from the
terminal working directory. Therefore the run is portable: after extracting the
archive, place the input files in Data and execute runAetherMMS.py from
any location.


Configuration
-------------

The configuration file is:

    config/config.txt

It uses simple key-value syntax:

    key = value

Comments are placed on the right of each value. The relevant scientific
parameters are:

    lidar_range_m
        Corridor half-width used to select urban objects around the trajectory.

    antenna_height_m
        Antenna phase-centre height above the local trajectory elevation.

    date_utc, start_time_utc, speed_kmh, step_sec
        Survey epoch definition. The main mission simulation is sampled at
        step_sec, normally 1 Hz.

    elevation_mask_deg
        Minimum satellite elevation angle used before obstruction testing.

    constellations
        Enabled GNSS constellations. Current release supports GPS and Galileo.

    max_ray_km
        Maximum search distance along each antenna-satellite ray for DTM,
        building and vegetation obstruction. The default is 1.2 km and is
        independent from lidar_range_m.

    selected_skyplot_mode
        Selection mode for the skyplot. Use time or distance.

    selected_skyplot_value
        If selected_skyplot_mode = time, this is seconds after mission start.
        If selected_skyplot_mode = distance, this is progressive route distance
        in metres. Leave empty to plot the middle epoch.


Computational workflow
----------------------

1. Urban scene construction - source/buildingcity.py

   The trajectory is read from KML and used as the spatial reference for the
   analysis corridor. DTM elevations are sampled from the GeoTIFF. Buildings and
   tree features are filtered against the route buffer. Tree crowns are stored
   with centre coordinates, crown radius, tree height and trunk radius. Optional
   GNSS bases are used to compute Base Check Coverage. Tunnel and covered-road
   geometries are imported from cache or retrieved from OpenStreetMap Overpass.

2. Satellite propagation - source/satellitepropagation.py

   GPS YUMA and Galileo XML almanacs are parsed and propagated for the requested
   UTC epochs. Satellite positions are transformed to local azimuth/elevation
   relative to the trajectory point.

3. Visibility classification - source/bestwindowsprediction.py

   For each satellite-epoch observation, AetherMMS applies the elevation mask
   and then tests the corresponding antenna-satellite ray against buildings,
   terrain and vegetation. The output class is LOS, NLOS, VEG or GNSS_DENIED.
   GNSS_DENIED is assigned for tunnel or covered-road segments.

4. Quality metrics

   Epoch-level metrics include visible satellite count, LOS satellite count,
   vegetation-degraded count, PDOP computed from LOS satellites, GPS/Galileo
   PDOP and Sky Visibility Index. Trajectory colours follow the convention:
   red for 0-4 LOS satellites or tunnel/covered road, orange for 5-7 LOS
   satellites, and green for more than 7 LOS satellites.

5. Best/worst temporal-window prediction

   The planning routine first evaluates a coarse set of candidate start times
   between planning_start_time_utc and planning_end_time_utc. It keeps the best
   and worst seeds, then refines each seed at -offset, 0 and +offset minutes.
   The final best and worst windows are selected from the refined candidates.
   The preview and refinement route sampling densities are controlled by:

       planning_preview_max_epochs
       planning_refinement_max_epochs

   These values are route samples used only for temporal planning. They are not
   satellites, not seconds and not the main 1 Hz mission epochs.


Output files
------------

CSV and metadata:

    results/csv/epochs.csv
    results/csv/satellite_visibility.csv
    results/csv/pdop_timeseries.csv
    results/csv/best_window_summary.csv
    results/csv/best_window_trajectory.csv
    results/csv/worst_window_trajectory.csv
    results/csv/metadata.txt

Figures:

    results/png/satellite_count_los_nlos_vegetation_present.png
    results/png/pdop_los_profile.png
    results/png/sky_visibility_index.png
    results/png/skyplot_selected_epoch.png
    results/png/trajectory_los_quality.png
    results/png/temporal_window_scores.png
    results/png/best_worst_temporal_windows.png

Report:

    results/AetherMMS_report.html

The report is an offline HTML document that references the PNG files in
results/png. It contains mission cards, base-coverage status, best/worst
temporal windows and all diagnostic figures.


Terminology
-----------

Satellite-epoch observations
    The number of satellite looks analysed by the engine. If an epoch contains
    18 satellites above the elevation mask, that epoch contributes 18
    satellite-epoch observations. Each observation corresponds to one
    antenna-satellite ray tested for obstruction.

Continuity
    Percentage of non-covered route samples in a temporal window that remain
    usable under the planning rule. A sample is considered non-continuous when
    LOS satellites are fewer than 4, PDOP is unavailable, or LOS PDOP exceeds
    8. Continuity equal to 100% means that no sampled point in that window
    violates those outage conditions.

Base Check Coverage
    Distance-based check against available GNSS bases, using the thresholds:
    green <= 5 km, yellow <= 10 km, orange <= 15 km and red > 15 km.


Numerical convention
--------------------

All exported quantities expressed in metres are rounded to at most 0.001 m.
Angles are reported in degrees where applicable. Survey times are UTC.


Browser-based viewer
--------------------

The webUI folder contains AetherMMS_v1.html, a standalone browser-based viewer
for interactive three-dimensional inspection of the scene, the trajectory and
the satellite analysis. It embeds an independent JavaScript engine that
implements the same algorithms as the Python core and produces matching
results; it is provided as an optional graphical aid for the user. The Python
core described above remains the reference implementation.


