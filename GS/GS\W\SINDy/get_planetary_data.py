import numpy as np

if False:
    import astropy as ap

    t = ap.time.Time("2026-04-01")
    mars = ap.coordinates.get_body("mars",t)
    print(mars)

if True:
    import numpy as np
    from astropy.time import Time
    from astropy.coordinates import get_body, get_body_barycentric, solar_system_ephemeris
    import astropy.units as u

    # Use JPL ephemerides for better accuracy
    solar_system_ephemeris.set('jpl')

    # Define time range
    start_time = Time("2026-01-01")
    end_time = Time("2026-01-10")

    # Create time array (1-day steps)
    num_points = 10
    times = start_time + np.linspace(0, (end_time - start_time).value, num_points) * u.day

    # Get Mars positions
    mars_positions = [(get_body("mars", t).icrs).cartesian.xyz.to(u.au) for t in times]
    mars_positions_2 = [get_body("mars",t) for t in times]
    print(mars_positions[0].value)
    print(mars_positions[1])
    print(mars_positions_2[0])
        
    # # Extract useful quantities
    # ra = [pos.ra.deg for pos in mars_positions]      # Right Ascension (degrees)
    # dec = [pos.dec.deg for pos in mars_positions]    # Declination (degrees)
    # distance = [pos.distance.au for pos in mars_positions]  # Distance (AU)
    
    # print(mars_positions[0].cartesian)
    # print(type(mars_positions[0].cartesian))
    # print(mars_positions[0].cartesian.x.to(u.au))


    # # Print results
    # for t, r, d, dist in zip(times.iso, ra, dec, distance):
    #     print(f"{t} | RA: {r:.3f} deg | Dec: {d:.3f} deg | Distance: {dist:.3f} AU")

def extract_planetary_data(planetName,nYears,tRes):
    solar_system_ephemeris.set('jpl')

    # Starting at the year 1900, extract a set of times at which to evaluate the planet's position
    startYear = 1900
    startTime = Time(f"{startYear}-01-01")
    endTime = Time(f"{startYear+nYears}-01-01")
    nPoints = round(nYears / tRes)
    times = start_time + np.linspace(0, (endTime - startTime).value, nPoints) * u.day

    # Obtain the planetary positions as an array of astropy data
    planet_positions = [(get_body(planetName,t).icrs).cartesian.xyz.to(u.au) for t in times]

    # Reshape into raw data in AU
    arr = np.array([]).reshape(0,3)
    for i in range(len(planet_positions)):
        arr = np.concatenate([arr,planet_positions[i].value[np.newaxis,:]],axis=0)
    return arr

def extract_multiplanetary_data(nYears, tRes, 
                                doPickle=True,
                                pickleFilename=None,
                                planets=["sun","mercury","venus","earth","mars","jupiter","saturn"]):

    # Extract data for each planet and combine into a single array
    pos = np.array([]).reshape(nYears * round(1/tRes),0)
    for planet in planets:
        planet_pos = extract_planetary_data(planet,nYears,tRes)
        pos = np.concatenate([pos,planet_pos],axis=1)
    
    if doPickle==True:
        import pickle
        with open(pickleFilename,"wb") as f:
            pickle.dump(pos,f)

    return pos
