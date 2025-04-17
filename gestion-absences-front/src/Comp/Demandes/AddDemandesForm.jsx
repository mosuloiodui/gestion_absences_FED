import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { Link , useNavigate} from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import {fetchMotifs} from '../../Features/Demande/MotifSlice';
import {addDemande} from '../../Features/Demande/DemandeSlice';


export default function AddDemandesForm() {
  
    const dispatch = useDispatch();
    const {motifs} = useSelector((state) => state.motifs);
    const { register, handleSubmit, watch, reset } = useForm();
    const [selectedMotif, setSelectedMotif] = useState(null);
    const navigate = useNavigate(); 

    useEffect(() => {
        dispatch(fetchMotifs());
    }, [dispatch]);

    const selectedMotifId = watch('motif');

    const handleMotifChange = (e) => {
      const selectedId = parseInt(e.target.value);
      const selectedOption = motifs.find(
        (motif) => motif.motifId === selectedId
      );
      setSelectedMotif(selectedOption);
    };

    const onSubmit = async (data) => {
      try {

        console.log(selectedMotif.motifId);
        const formData = new FormData();

  formData.append('motif', new Blob([JSON.stringify({'motif':selectedMotif.motifId})], { type: 'application/json'}) );  
   // formData.append('dateDepart', data.dateDepart);
    //formData.append('dateReprise', data.dateReprise);
    //formData.append('observations', data.observations);
    formData.append('file', data.file[0]); 
   // formData.append('ppr', data.ppr);
        dispatch(addDemande(formData));
        reset();
        setSelectedMotif(null); 
     /*   if (window.confirm("l'employé a été ajouté avec succès!")) {
          navigate('/ListDemandes');
          }*/
      } catch (error) {
        console.error('Error saving demande:', error);
      }
    };

    return (
      <div className = "container">
    	<div className = "row">
    	<div className = "col-md-8 mx-auto rounded border p-4 m-4">
    		<h2 class = "text-center mb-5">Ajouter une Demande</h2>
        
        <form onSubmit={handleSubmit(onSubmit)} encType="multipart/form-data">

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">PPR:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input className="form-control"{...register('ppr', { required: "Veuillez saisir le PPR" })}  />
              </div>
        </div>

        <div className = "row mb-3">
    					<label className="col-sm-4 col-form-label">Motif<span className="text-danger">*</span></label>
    					<div className="col-sm-8">
    						<select {...register('motif')} required className="form-control"
                onChange={handleMotifChange}>       
    							<option value ="">-- Sélectionner un Motif --</option>
    							{motifs.map((motif) => (
                    <option key={motif.motifId} value={motif.motifId}>
                        {motif.motifLibelle}
                    </option>
                ))}
    						</select>		
    					</div>
    	</div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Date Départ:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input type="date" className="form-control"{...register('dateDepart')} required />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Date Reprise:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input type="date" className="form-control"{...register('dateReprise')} required />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Pièce justificative:<span className="text-danger">*</span></label>
              <div className="col-sm-8">
                <input type="file"  className="form-control"{...register('file')} />
              </div>
        </div>

        <div className = "row mb-3">
            <label className="col-sm-4 col-form-label">Observations:</label>
              <div className="col-sm-8">
                <textarea  className="form-control"{...register('observations')} />
              </div>
        </div>
        

            <div className="row">
    					<div className = "offset-sm-4 col-sm-4 d-grid">
    						<button type = "submit" className = "btn btn-primary"> Enregistrer </button>
    					</div>
    					<div className = "col-sm-4 d-grid">
    						<Link to ='/listDemandes'className="btn btn-outline-primary" role="button"> Annuler </Link>
    					</div>
    				</div>
        </form>
      </div>
    </div>
    </div>
    );
};


