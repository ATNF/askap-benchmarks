/// @file DataSet.cc
///
/// @copyright (c) 2009 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include own header file first
#include "DataSet.h"

// System includes
#include <string>
#include <iostream>
#include <ctime>

// ASKAPsoft includes
#include <Common/ParameterSet.h>
#include <casa/Arrays/Matrix.h>
#include <tables/Tables/TableDesc.h>
#include <tables/Tables/SetupNewTab.h>
#include <tables/Tables/IncrementalStMan.h>
#include <tables/Tables/StandardStMan.h>
#include <tables/Tables/TiledShapeStMan.h>
#include <ms/MeasurementSets/MSColumns.h>

using namespace casa;
using LOFAR::ParameterSet;

DataSet::DataSet(const std::string& filename, const LOFAR::ParameterSet& parset)
: itsParset(parset)
{
    create(filename);
    initAnt();
    initFields();
    initSpWindows();
    initFeeds();
    initObs();
}

DataSet::~DataSet()
{
    delete itsMs;
}

void DataSet::add(void)
{
    MSColumns msc(*itsMs);

    const int intTime = itsParset.getInt32("integrationTime"); 
    const int nAnt = itsParset.getInt32("nAntenna");
    const int nChan = itsParset.getInt32("nChan");
    const int nCorr = itsParset.getInt32("nPol");
    const int nFeeds = itsParset.getInt32("nFeeds");
    const int nBaselines = nAnt * (nAnt + 1) / 2;

    // Save row cursor
    int row = itsMs->nrow();

    itsMs->addRow(nFeeds*nBaselines);

    Matrix<Complex> data(nCorr,nChan);
    data.set(Complex(0.0));

    Matrix<Bool> flag(nCorr,nChan);
    flag = False;

    for (int feed = 0; feed < nFeeds; ++feed) {

        // Do the first row per feed outside the loop,
        // the values are maintained for each row until
        // another put().
        msc.scanNumber().put(row, 0);
        msc.fieldId().put(row, 0);
        msc.dataDescId().put(row, 0);
        msc.time().put(row, 0 );
        msc.arrayId().put(row, 0);
        msc.processorId().put(row, 0);
        msc.exposure().put(row, intTime);
        msc.interval().put(row, intTime);
        msc.observationId().put(row, 0);
        msc.stateId().put(row, -1);


        for (int ant1 = 0; ant1 < nAnt; ++ant1) {
            const int startAnt = ant1;
            for (int ant2 = startAnt; ant2 < nAnt; ++ant2) {
                msc.antenna1().put(row,ant1);
                msc.antenna2().put(row,ant2);
                msc.feed1().put(row,feed);
                msc.feed2().put(row,feed);

                Vector<double> uvwvec(3);
                uvwvec(0) = 1;
                uvwvec(1) = 2;
                uvwvec(2) = 3;
                msc.uvw().put(row,uvwvec);

                msc.data().put(row, data);
                msc.flag().put(row, flag);
                msc.flagRow().put(row, False);

                Vector<Float> weight(nCorr);
                weight = 4.0;
                Vector<Float> sigma(nCorr);
                sigma = 5.0;
                msc.weight().put(row, weight);
                msc.sigma().put(row, sigma);

                row++;
            } // Ant2
        } // Ant1
    } // Feed

    // Add pointing
    int pointingRow = itsMs->pointing().nrow();
    itsMs->pointing().addRow(nAnt);
    MSPointingColumns& pointingc=msc.pointing();
    Vector<MDirection> direction(1);

    pointingc.numPoly().put(pointingRow, 0);
    pointingc.interval().put(pointingRow, -1);
    pointingc.tracking().put(pointingRow, True);
    pointingc.time().put(pointingRow, 0);
    pointingc.timeOrigin().put(pointingRow, 0);
    pointingc.interval().put(pointingRow, 0);
    pointingc.antennaId().put(pointingRow, 0);
    pointingc.directionMeasCol().put(pointingRow, direction);
    pointingc.targetMeasCol().put(pointingRow, direction);
}

void DataSet::create(const std::string& filename)
{
    int bucketSize = itsParset.getInt32("stman.bucketsize");
    if (bucketSize < 8192) {
        bucketSize = 8192;
    }
    int tileNcorr = itsParset.getInt32("stman.tilencorr");
    if (tileNcorr <= 1) {
        tileNcorr = 1;
    }
    int tileNchan = itsParset.getInt32("stman.tilenchan");
    if (tileNchan <= 1) {
        tileNchan = 1;
    }

    std::cout << "Creating dataset " << filename << std::endl;

    // Make MS with standard columns
    TableDesc msDesc(MS::requiredTableDesc());

    // Add the DATA column.
    MS::addColumnToDesc(msDesc, MS::DATA, 2);

    SetupNewTable newMS(filename, msDesc, Table::New);

    // Set the default Storage Manager to be the Incr one
    {
        IncrementalStMan incrStMan("ismdata", bucketSize);
        newMS.bindAll(incrStMan, True);
    }

    // Bind ANTENNA1, and ANTENNA2 to the standardStMan 
    // as they may change sufficiently frequently to make the
    // incremental storage manager inefficient for these columns.

    {
        StandardStMan ssm("ssmdata", bucketSize);
        newMS.bindColumn(MS::columnName(MS::ANTENNA1), ssm);
        newMS.bindColumn(MS::columnName(MS::ANTENNA2), ssm);
        newMS.bindColumn(MS::columnName(MS::UVW), ssm);
    }

    // These columns contain the bulk of the data so save them in a tiled way
    {
        // Get nr of rows in a tile.
        int nrowTile = std::max(1, bucketSize / (8*tileNcorr*tileNchan));
        TiledShapeStMan dataMan("TiledData",
                IPosition(3,tileNcorr, tileNchan,nrowTile));
        newMS.bindColumn(MeasurementSet::columnName(MeasurementSet::DATA),
                dataMan);
        newMS.bindColumn(MeasurementSet::columnName(MeasurementSet::FLAG),
                dataMan);
    }
    {
        int nrowTile = std::max(1, bucketSize / (4*8));
        TiledShapeStMan dataMan("TiledWeight",
                IPosition(2,4,nrowTile));
        newMS.bindColumn(MeasurementSet::columnName(MeasurementSet::SIGMA),
                dataMan);
        newMS.bindColumn(MeasurementSet::columnName(MeasurementSet::WEIGHT),
                dataMan);
    }

    // Now we can create the MeasurementSet and add the (empty) subtables
    itsMs = new MeasurementSet(newMS,0);
    itsMs->createDefaultSubtables(Table::New);
    itsMs->flush();

    // Set the TableInfo
    {
        TableInfo& info(itsMs->tableInfo());
        info.setType(TableInfo::type(TableInfo::MEASUREMENTSET));
        info.setSubType(String("simulator"));
        info.readmeAddLine("This is a MeasurementSet Table holding simulated astronomical observations");
    }
}

void DataSet::initAnt(void)
{
    const int nAnt = itsParset.getInt32("nAntenna");

    MSColumns msc(*itsMs);
    MSAntenna& ant = itsMs->antenna();
    ant.addRow(nAnt);
}

void DataSet::initFields(void)
{
    const int nFields = itsParset.getInt32("nFields");

    //MSColumns msc(*itsMs);
    MSField& field = itsMs->field();
    field.addRow(nFields);
}

void DataSet::initSpWindows(void)
{
    itsMs->spectralWindow().addRow(1);
    itsMs->polarization().addRow(1);
    itsMs->dataDescription().addRow(1);
}

void DataSet::initFeeds(void)
{
    const int nFeeds = itsParset.getInt32("nFeeds");
    itsMs->feed().addRow(nFeeds);
}

void DataSet::initObs(void)
{
    MSObservation& obs = itsMs->observation();
    obs.addRow();
}
