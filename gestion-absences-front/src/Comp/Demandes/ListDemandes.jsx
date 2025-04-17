import React, { useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {fetchDemandes,deleteDemande} from '../../Features/Demande/DemandeSlice';
import { setSelectedDemande } from '../../Features/Demande/DemandeSlice'; 
import {useNavigate} from 'react-router-dom';

export default function ListDemandes() {

const dispatch = useDispatch();
const data = useSelector((state) => state.demandes.list);
const status = useSelector((state) => state.demandes.status);
const error = useSelector((state) => state.demandes.error);
const navigate = useNavigate(); 

	  
useEffect(() => {
	if (status === 'idle') {
		dispatch(fetchDemandes());
		}
}, [dispatch, status]);

const handleDelete = async(id) => {
	if (window.confirm("Êtes-vous sûr de vouloir supprimer cette demande ?")) {
	 dispatch(deleteDemande(id)); 
	dispatch(fetchDemandes()); 
	}
  };
	  
if (status === 'loading') {
	return <div>Loading...</div>;
	}
	  
if (status === 'failed') {
	return <div>Error: {error}</div>;
}

  return (
    
    <div className = "container" >
    <h5 style={{margin: '25px 0px 30px 20px'}}>Liste des Demandes</h5>
    	<table className="table table-striped" style={{width:'100%'}}>
            <thead>
    			<tr>
    				<th>PPR</th>
    				<th>Nom</th>
    				<th>Prenom</th>
    				<th>Motif d'absence</th>
    				<th>Durée (par journée)</th>
    				<th>Date de création</th>
    				<th>Action</th>
    			</tr>		
    		</thead>
    		<tbody>
                {
                    data.map(demande =>
    			    <tr key ={demande.demandeId}>
    				    <td>{demande.employeDto.ppr}</td>
    				    <td>{demande.employeDto.nom}</td>
    			        <td>{demande.employeDto.prenom}</td>
                        <td>{demande.motifDto.motifLibelle}</td>
                        <td>{demande.duree}</td>
                        <td>{demande.date_creation}</td>
    				    <td style ={{whiteSpace:'nowrap'}}>
    					    <a className="btn btn-primary btn-sm"> Modifier</a>			 					
    					    <button onClick={() => handleDelete(demande.demandeId)} className="btn btn-primary btn-sm" > Supprimer</button> 		
    				    </td>
    			    </tr>)
                }
    		</tbody>
    	
    	</table>
    </div>

  )
}
