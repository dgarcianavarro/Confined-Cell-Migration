/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"
#include <vector>


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <math.h>
#include <string>



void create_cell_types( void )
{
	// set the random seed 
	SeedRandom( parameters.ints("random_seed") );  
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	//cell_defaults.functions.update_velocity = standard_update_cell_velocity;
	cell_defaults.functions.update_velocity = velocity_modification;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/*
       Cell rule definitions 
	*/

	setup_cell_rules(); 

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.contact_function = contact_function; 
	
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	display_cell_definitions( std::cout ); 
	
	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	
	return; 
}

void setup_tissue( void )
{
	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell 
	
	Cell* pC;
	
	for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	{
		Cell_Definition* pCD = cell_definitions_by_index[k]; 
		std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
		for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
		{
			std::vector<double> position = {0,0,0}; 
			position[0] = Xmin + UniformRandom()*Xrange; 
			position[1] = Ymin + UniformRandom()*Yrange; 
			position[2] = Zmin + UniformRandom()*Zrange; 
			
			pC = create_cell( *pCD ); 
			pC->assign_position( position );
		}
	}
	std::cout << std::endl; 
	
	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml(); 	
	
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{ return paint_by_number_cell_coloring(pCell); }


void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 



void velocity_modification( Cell* pCell, Phenotype& phenotype , double dt )
{
	double dyn_viscosity= parameters.doubles("Dyn_viscosity");
	double R_p = pCell->phenotype.geometry.radius;
	bool is_membrana = 0;
	bool is_indentador = 0; 
	bool movil=0;
	double viscosidad = parameters.doubles("viscosity"); 
	double f_locom = parameters.doubles("F_loc");
	double current_time;
	int Cell_Type;
	int Cell_ID;
	current_time = PhysiCell_globals.current_time;
	Cell_Type = pCell->type;
	Cell_ID = pCell->ID;

	if(Cell_Type==0)
	{
		is_membrana	= 0;
		movil		= 1;
	}
	else if (Cell_Type==1)
	{
		is_membrana		= 0;
		movil			= 1;
	}
	else if(Cell_Type == 2)
	{
		is_membrana = 1;
		movil 		= 1;
	}
	else if(Cell_Type == 3)
	{
		is_membrana = 0;
		movil		= 0;
		pCell->is_movable = false; 

	}
	else if(Cell_Type == 4)
	{
		is_membrana = 0;
		if (current_time <4)
		{
			movil		= 0;
		}
		if (current_time >=4)
		{
			movil		= 1;
		}
		is_indentador	= 1;


	}


	// Velocity update using the standard method
	standard_update_cell_velocity(pCell, phenotype, dt);

	std::vector<double> F_k (3, 0);
	std::vector<double> F_loc (3, 0);
	double F_v_virtual;
	double F_drag_virtual;

	
	F_k[0] = pCell->velocity[0];
	F_k[1] = pCell->velocity[1];
	F_k[2] = pCell->velocity[2];
	F_v_virtual=viscosidad/60; 
	F_drag_virtual = 1*1e-7*3.141592*R_p*dyn_viscosity*is_membrana;

	if (current_time <= 4)
	{
		F_loc [0] = 0;
		F_loc [1] = f_locom;
		F_loc [2] = 0;

	}
	if (current_time > 4)
	{
		F_loc [0] = 0;
		F_loc [1] = 0;
		F_loc [2] = 0;
	}

	pCell->velocity [0] += F_loc[0];   // Fk+Fg
	pCell->velocity [1] += F_loc[1];   // Fk+Fg
	pCell->velocity [2] += F_loc[2];   // Fk+Fg

	pCell->velocity [0] /=(F_v_virtual+F_drag_virtual);	//v=(Fk+Fg)/(Fv/v+Fdrag/v)
	pCell->velocity [1] /=(F_v_virtual+F_drag_virtual);	//v=(Fk+Fg)/(Fv/v+Fdrag/v)
	pCell->velocity [2] /=(F_v_virtual+F_drag_virtual);	//v=(Fk+Fg)/(Fv/v+Fdrag/v)
	pCell->velocity [0]	*= movil;
	pCell->velocity [1]	*= movil;
	pCell->velocity [2]	*= movil;


	// We calculate the actual Fv and Fdrag to have that information
	std::vector<double> F_v (3, 0);
	F_v [0]	= F_v_virtual * pCell->velocity [0];
	F_v [1]	= F_v_virtual * pCell->velocity [1];
	F_v [2]	= F_v_virtual * pCell->velocity [2];

	std::vector<double> F_drag (3, 0);
	F_drag [0]	= F_drag_virtual * pCell->velocity [0];
	F_drag [1]	= F_drag_virtual * pCell->velocity [1];
	F_drag [2]	= F_drag_virtual * pCell->velocity [2];

	
	if (Cell_Type == 4 && fmod(current_time, 0.01) < 0.00001 && current_time >4 && F_k[0] != 0)
	{
		saveForcesDesp(current_time, Cell_ID, pCell->position[0], F_k[0], "output/forcesDesp.csv");

	}
	
	if (current_time <=4.05 && pCell->type == 4) // Surface
	{
		pCell->velocity [0] = 0; 
		pCell->velocity [1] = 0;
		pCell->velocity [2] = 0;
	}
	else if (current_time >4.05 && pCell->type == 4) // Surface
	{
		pCell->velocity [0] = -0.01;//-5*0.066;
	  	pCell->velocity [1] = 0; 
		pCell->velocity [2] = 0;

	}

	// We set the velocity to zero for non-movable particles
	pCell->velocity [0]	*= movil;
	pCell->velocity [1]	*= movil;
	pCell->velocity [2]	*= movil;

return; 
} 



std::vector<std::string> custom_coloring_function( Cell* pCell)
{

	int Cell_Type = pCell->type;

	std::vector<std::string> output = paint_by_number_cell_coloring(pCell);
	if( Cell_Type==0 ) // Cyto
	{
		char szColor[1024];

		sprintf( szColor, "rgb(%u,%u,%u)",144,238,144 );
		output[0] = szColor;
		output[2] = szColor;
		output[3] = szColor;
	}
	if( Cell_Type==1 ) // Nucleus
	{
		char szColor[1024];

		sprintf( szColor, "rgb(%u,%u,%u)",205,133,63 );
		output[0] = szColor;
		output[2] = szColor;
		output[3] = szColor;
	}
	if( Cell_Type==2 ) // Membrane
	{
		char szColor[1024];

		sprintf( szColor, "rgb(%u,%u,%u)",0,128,0 );
		output[0] = szColor;
		output[2] = szColor;
		output[3] = szColor;
	}
	if( Cell_Type==3 ) // Surface
	{
		char szColor[1024];

		sprintf( szColor, "rgb(%u,%u,%u)",128,128,128 );
		output[0] = szColor;
		output[2] = szColor;
		output[3] = szColor;
	}
	if( Cell_Type==4 ) // Indentator
	{
		char szColor[1024];
		sprintf( szColor, "rgb(%u,%u,%u)",178,34,34 );
		output[0] = szColor;
		output[2] = szColor;
		output[3] = szColor;
	}
return output;
}



void saveForcesDesp(double current_time, int ID, double posx, double force, const std::string& filename) {
	// Check if the file already exists before opening it
    bool fileExists = std::ifstream(filename).good();
    
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;

    // Write the header only if the file is new

    if (!fileExists) {
        file << "current_time,ID,posx,force\n";
    }

    file << current_time << "," << ID << "," << posx << "," << force << "\n";
    file.close();
}